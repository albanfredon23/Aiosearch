"""
Benchmark reproductible en une commande, sur CPU, sans réseau ni clé d'API :

    python -m aiotech.bench [--datasets hotpotqa,fever] [--limit N] [--llm] [--out DOSSIER]
    docker compose run --rm bench

Sans --llm : mesures hors ligne (recherche des preuves, réponse vérifiée, abstention, coût de calcul).
Avec --llm (ANTHROPIC_API_KEY ou configuration LiteLLM) : en plus, EM/F1 HotpotQA et exactitude FEVER
avec le même lecteur LLM pour les deux systèmes, jetons et coût par requête.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

from aiotech import __version__
from aiotech.bench.datasets import FeverSet, HotpotItem, load_fever, load_hotpot
from aiotech.bench.metrics import (
    all_at,
    bootstrap,
    contains_answer,
    exact_match,
    f1_score,
    paired_difference,
    recall_at,
    reciprocal_rank,
)
from aiotech.bench.reader import Reader
from aiotech.bench.resources import DEFAULT_CPU_WATTS, Meter, PythonPeak, host_info, peak_rss_mb
from aiotech.bench.systems import Aiotech, ClassicRag, build_store
from aiotech.llm.base import LLMError
from aiotech.models import Passage, SearchResult
from aiotech.runtime import build_routers
from aiotech.settings import Settings

SEED = 20261006
MEMORY_SAMPLE = 20
SYSTEM_LABELS = {"rag": "RAG classique (BM25 top-k)", "aiotech": "AIOTECH Search (auto)",
                 "aiotech-deep": "AIOTECH Search (approfondi)"}

log = logging.getLogger("aiotech.bench")
T = TypeVar("T")


@dataclass
class Collector:
    """Valeurs par question et par système ; une question en erreur est retirée pour tous (comparaison appariée)."""

    systems: tuple[str, ...]
    values: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    meters: dict[str, Meter] = field(default_factory=dict)
    errors: Counter[str] = field(default_factory=Counter)
    skipped: int = 0

    def __post_init__(self) -> None:
        self.meters = {s: Meter() for s in self.systems}

    def add(self, row: dict[str, dict[str, float]]) -> None:
        for metric, per_system in row.items():
            table = self.values.setdefault(metric, {s: [] for s in self.systems})
            for system, value in per_system.items():
                table[system].append(value)


def doc_ranking(passages: Sequence[Passage]) -> list[str]:
    return list(dict.fromkeys(p.document_id for p in passages))


async def timed(meter: Meter, work: Awaitable[T]) -> T:
    started = meter.start()
    try:
        return await work
    finally:
        meter.stop(started)


def timed_sync(meter: Meter, work: Callable[[], T]) -> T:
    started = meter.start()
    try:
        return work()
    finally:
        meter.stop(started)


def possibility_texts(result: SearchResult) -> list[str]:
    return [f"{p.answer} (verified, reliability {p.reliability_pct}%)" for p in result.possibilities]


async def hotpot_item(item: HotpotItem, col: Collector, reader: Reader | None) -> None:
    store = build_store(item.documents())
    titles = {d.id: d.title for d in item.documents()}
    rag, auto, deep = ClassicRag(store), Aiotech(store, "auto"), Aiotech(store, "deep")
    gold = set(item.supporting_titles)

    def rag_answer() -> str | None:
        rag.retrieve(item.question, 5)
        return rag.top_sentence(item.question)[0]

    rag_sentence = timed_sync(col.meters["rag"], rag_answer)
    results: dict[str, SearchResult] = {}
    for engine in (auto, deep):
        results[engine.name] = await timed(col.meters[engine.name], engine.search(item.question))

    rankings = {
        "rag": [titles[d] for d in doc_ranking(rag.retrieve(item.question, 10))],
        "aiotech": [titles[d] for d in doc_ranking(auto.retrieve(item.question, 10))],
        "aiotech-deep": [titles[d] for d in doc_ranking(deep.retrieve(item.question, 10))],
    }
    row: dict[str, dict[str, float]] = {
        "sp_recall@2": {s: recall_at(r, gold, 2) for s, r in rankings.items()},
        "sp_recall@5": {s: recall_at(r, gold, 5) for s, r in rankings.items()},
        "sp_all@2": {s: all_at(r, gold, 2) for s, r in rankings.items()},
        "sp_all@5": {s: all_at(r, gold, 5) for s, r in rankings.items()},
    }
    answers = {"rag": rag_sentence, **{name: Aiotech.top_answer(res)[0] for name, res in results.items()}}
    row["answered"] = {s: float(a is not None) for s, a in answers.items()}
    if not item.yes_no:
        row["answer_in_top"] = {s: contains_answer(a, item.answer) for s, a in answers.items()}
        row["answer_in_top_when_answered"] = {s: contains_answer(a, item.answer) for s, a in answers.items()
                                              if a is not None}
    if reader is not None:
        evidence = {
            "rag": [p.text for p in rag.retrieve(item.question, 5)],
            "aiotech": possibility_texts(results["aiotech"]) + [p.text for p in auto.retrieve(item.question, 5)],
            "aiotech-deep": possibility_texts(results["aiotech-deep"]) + [p.text for p in deep.retrieve(item.question, 5)],
        }
        em: dict[str, float] = {}
        f1: dict[str, float] = {}
        for system, texts in evidence.items():
            prediction, completion, latency = await reader.answer(item.question, texts)
            col.meters[system].add_llm(completion.tokens_in, completion.tokens_out, completion.cost_usd, latency)
            em[system], f1[system] = exact_match(prediction, item.answer), f1_score(prediction, item.answer)
        row["llm_em"], row["llm_f1"] = em, f1
    col.add(row)


async def fever_item(claim_index: int, fever: FeverSet, rag: ClassicRag, auto: Aiotech, col: Collector,
                     reader: Reader | None) -> None:
    claim = fever.claims[claim_index]
    gold = {claim.evidence_id}
    rag_rank = timed_sync(col.meters["rag"], lambda: doc_ranking(rag.retrieve(claim.claim, 10)))
    result = await timed(col.meters["aiotech"], auto.search(claim.claim))
    aio_rank = doc_ranking(auto.retrieve(claim.claim, 10))
    top_answer, top_sources = Aiotech.top_answer(result)
    row: dict[str, dict[str, float]] = {
        "evidence_recall@1": {"rag": recall_at(rag_rank, gold, 1), "aiotech": recall_at(aio_rank, gold, 1)},
        "evidence_recall@5": {"rag": recall_at(rag_rank, gold, 5), "aiotech": recall_at(aio_rank, gold, 5)},
        "evidence_mrr@10": {"rag": reciprocal_rank(rag_rank, gold), "aiotech": reciprocal_rank(aio_rank, gold)},
        "top_answer_cites_gold": {"rag": float(rag_rank[:1] == [claim.evidence_id]),
                                  "aiotech": float(claim.evidence_id in top_sources[:1])},
        "answered": {"rag": float(bool(rag_rank)), "aiotech": float(top_answer is not None)},
    }
    if reader is not None:
        evidence = {
            "rag": [fever.evidence[d] for d in rag_rank[:5]],
            "aiotech": possibility_texts(result) + [fever.evidence[d] for d in aio_rank[:5]],
        }
        accuracy: dict[str, float] = {}
        for system, texts in evidence.items():
            label, completion, latency = await reader.label(claim.claim, texts)
            col.meters[system].add_llm(completion.tokens_in, completion.tokens_out, completion.cost_usd, latency)
            accuracy[system] = float(label == claim.label)
        row["llm_label_accuracy"] = accuracy
    col.add(row)


async def guarded(col: Collector, work: Awaitable[None]) -> None:
    try:
        await work
    except Exception as exc:
        col.errors[exc.__class__.__name__] += 1
        col.skipped += 1
        log.warning("question ignorée après erreur %s : %s", exc.__class__.__name__, exc)


async def python_peaks(names: Sequence[str], calls: dict[str, Callable[[], Awaitable[Any]]]) -> dict[str, float]:
    peaks: dict[str, float] = {}
    for name in names:
        with PythonPeak() as peak:
            try:
                await calls[name]()
            except Exception as exc:
                log.warning("mesure mémoire %s impossible : %s", name, exc)
        peaks[name] = peak.peak_mb
    return peaks


METRIC_LABELS = {
    "sp_recall@2": "Rappel des paragraphes utiles @2",
    "sp_recall@5": "Rappel des paragraphes utiles @5",
    "sp_all@2": "Les 2 paragraphes utiles dans le top 2",
    "sp_all@5": "Les 2 paragraphes utiles dans le top 5",
    "answered": "Taux de réponse (sinon abstention)",
    "answer_in_top": "Réponse attendue dans la réponse affichée en tête (hors oui/non)",
    "answer_in_top_when_answered": "Idem, parmi les questions où le système répond",
    "llm_em": "Exact Match (lecteur LLM commun)",
    "llm_f1": "F1 (lecteur LLM commun)",
    "evidence_recall@1": "Preuve exacte en 1re position",
    "evidence_recall@5": "Preuve exacte dans le top 5",
    "evidence_mrr@10": "MRR@10 de la preuve exacte",
    "top_answer_cites_gold": "Réponse de tête appuyée sur la preuve exacte",
    "llm_label_accuracy": "Exactitude SUPPORTS/REFUTES (lecteur LLM commun)",
}


def summarize(col: Collector, cpu_watts: float, peaks: dict[str, float]) -> dict[str, Any]:
    metrics = []
    for metric, table in col.values.items():
        entry: dict[str, Any] = {"metric": metric, "label": METRIC_LABELS.get(metric, metric), "values": {}}
        for system, values in table.items():
            entry["values"][system] = {**bootstrap(values, SEED).as_dict(), "n": len(values)}
        base = table.get("rag", [])
        for system, values in table.items():
            if system != "rag" and len(values) == len(base) and values:
                entry.setdefault("difference_vs_rag", {})[system] = paired_difference(values, base, SEED).as_dict()
        metrics.append(entry)
    return {
        "metrics": metrics,
        "efficiency": {s: m.summary(cpu_watts, peaks.get(s)) for s, m in col.meters.items()},
        "errors": dict(col.errors),
        "skipped": col.skipped,
    }


async def run_hotpot(limit: int | None, reader: Reader | None, cpu_watts: float) -> dict[str, Any]:
    items = load_hotpot(limit)
    col = Collector(("rag", "aiotech", "aiotech-deep"))
    for item in items:
        await guarded(col, hotpot_item(item, col, reader))
    sample = items[:MEMORY_SAMPLE]

    async def run_rag() -> None:
        for item in sample:
            rag = ClassicRag(build_store(item.documents()))
            rag.retrieve(item.question, 5)
            rag.top_sentence(item.question)

    async def run_aiotech(depth: str) -> None:
        for item in sample:
            await Aiotech(build_store(item.documents()), "deep" if depth == "deep" else "auto").search(item.question)

    peaks = await python_peaks(col.systems, {"rag": run_rag, "aiotech": lambda: run_aiotech("auto"),
                                             "aiotech-deep": lambda: run_aiotech("deep")})
    summary = summarize(col, cpu_watts, peaks)
    summary.update({
        "dataset": "HotpotQA (dev, distractor)",
        "n": len(items) - col.skipped,
        "sampling": "300 questions tirées sans remise, random.Random(20261006), parmi les 7 405 du dev distractor officiel",
        "setting": "10 paragraphes par question (2 utiles, 8 leurres) ; titre + texte indexés à l'identique",
        "yes_no_excluded_from_answer_metrics": sum(1 for i in items if i.yes_no),
    })
    return summary


async def run_fever(limit: int | None, reader: Reader | None, cpu_watts: float) -> dict[str, Any]:
    fever = load_fever(limit)
    store = build_store(fever.documents())
    rag, auto = ClassicRag(store), Aiotech(store, "auto")
    col = Collector(("rag", "aiotech"))
    for index in range(len(fever.claims)):
        await guarded(col, fever_item(index, fever, rag, auto, col, reader))
    sample = fever.claims[:MEMORY_SAMPLE]

    async def run_rag() -> None:
        for c in sample:
            rag.retrieve(c.claim, 10)

    async def run_aiotech() -> None:
        fresh = Aiotech(store, "auto")
        for c in sample:
            await fresh.search(c.claim)

    peaks = await python_peaks(col.systems, {"rag": run_rag, "aiotech": run_aiotech})
    summary = summarize(col, cpu_watts, peaks)
    labels = Counter(c.label for c in fever.claims)
    summary.update({
        "dataset": "FEVER (dev, paires affirmation / preuve originales)",
        "n": len(fever.claims) - col.skipped,
        "sampling": "les 355 paires originales du dev FEVER publiées par FeverSymmetric v0.2 (dev + test), sans tirage",
        "setting": (f"corpus commun de {len(fever.evidence)} phrases : les preuves originales et "
                    f"{len(fever.synthetic)} phrases modifiées (leurres difficiles) ; labels {dict(labels)}"),
    })
    return summary


def fmt(value: dict[str, float], pct: bool = True) -> str:
    scale = 100.0 if pct else 1.0
    return f"{value['mean'] * scale:.1f} [{value['low'] * scale:.1f} ; {value['high'] * scale:.1f}]"


def markdown(report: dict[str, Any]) -> str:
    lines = [f"# Benchmark AIOTECH Search {report['version']}", "",
             f"Exécuté le {report['generated_at']} sur {report['host']['platform']} "
             f"({report['host']['cpu_count']} cœurs, Python {report['host']['python']}), CPU uniquement.",
             f"Lecteur LLM : {report['llm']['status']}.", ""]
    for data in report["datasets"].values():
        systems = list(data["efficiency"])
        lines += [f"## {data['dataset']} : n = {data['n']}", "", f"Échantillon : {data['sampling']}.",
                  f"Cadre : {data['setting']}.", "",
                  "Valeurs en % avec intervalle de confiance à 95 % (bootstrap apparié, 2 000 tirages).", "",
                  "| Mesure | " + " | ".join(SYSTEM_LABELS[s] for s in systems) + " | Écart AIOTECH − RAG |",
                  "|---|" + "---|" * (len(systems) + 1)]
        for m in data["metrics"]:
            cells = [fmt(m["values"][s]) if s in m["values"] and m["values"][s]["n"] else "—" for s in systems]
            diff = m.get("difference_vs_rag", {}).get("aiotech")
            lines.append(f"| {m['label']} | " + " | ".join(cells) + f" | {fmt(diff) if diff else '—'} |")
        lines += ["", "| Coût par requête | " + " | ".join(SYSTEM_LABELS[s] for s in systems) + " |",
                  "|---|" + "---|" * len(systems)]
        eff = data["efficiency"]
        rows: list[tuple[str, Callable[[dict[str, Any]], str]]] = [
            ("Latence p50 (ms)", lambda e: f"{e['latency_ms']['p50']:.1f}"),
            ("Latence p95 (ms)", lambda e: f"{e['latency_ms']['p95']:.1f}"),
            ("Temps CPU (ms)", lambda e: f"{e['cpu_ms_per_query']:.1f}"),
            ("Énergie (mWh)", lambda e: f"{e['energy_per_query']['value_mwh']:.4f}"),
            ("Pic mémoire Python (Mo, 20 requêtes)", lambda e: f"{e['python_peak_mb']}"),
            ("VRAM (Mo)", lambda e: "0"),
            ("Appels LLM", lambda e: str(e["llm_calls"])),
            ("Jetons entrée / sortie", lambda e: f"{e['tokens_in_per_query']} / {e['tokens_out_per_query']}"),
            ("Coût API (USD)", lambda e: "inconnu" if e["cost_usd_per_query"] is None else f"{e['cost_usd_per_query']:.6f}"),
        ]
        for label, render in rows:
            lines.append(f"| {label} | " + " | ".join(render(eff[s]) for s in systems) + " |")
        method = next(iter(eff.values()))["energy_per_query"]["method"]
        lines += ["", f"Énergie : {method}. Latence AIOTECH = pipeline complet (recherche, graphe, SCG/TAP, "
                      "vérification) ; latence RAG = recherche BM25 et choix de la phrase de tête, sans LLM.",
                  f"Erreurs : {data['errors'] or 'aucune'}.", ""]
    lines += [f"Pic mémoire du processus (RSS) : {report['process']['peak_rss_mb']} Mo. VRAM : 0 (aucun calcul GPU). "
              "FLOPs d'un LLM distant : non mesurables côté client, non publiés.", ""]
    return "\n".join(lines)


def write_outputs(out: Path, report: dict[str, Any], text: str) -> list[str]:
    try:
        out.mkdir(parents=True, exist_ok=True)
        (out / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
        (out / "results.md").write_text(text, "utf-8")
    except OSError as exc:
        return [f"résultats non écrits dans {out} ({exc.strerror}) : le rapport ci-dessus fait foi"]
    return [f"résultats écrits dans {out}/results.json et {out}/results.md"]


def make_reader(enabled: bool, model: str | None) -> tuple[Reader | None, str]:
    if not enabled:
        return None, "désactivé (mesures hors ligne ; ajouter --llm et une clé pour EM/F1 et exactitude)"
    try:
        settings = Settings.from_env()
    except ValueError as exc:
        return None, f"indisponible ({exc}) : mesures hors ligne seulement"
    if model:
        settings = replace(settings, llm_model=model)
    router, fast, notes = build_routers(settings)
    chosen = fast or router
    if chosen is None:
        return None, f"indisponible ({'; '.join(notes)}) : mesures hors ligne seulement"
    return Reader(chosen), f"{chosen.primary_model} (même consigne et même budget de preuves pour tous les systèmes)"


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    reader, status = make_reader(args.llm, args.llm_model)
    datasets: dict[str, Any] = {}
    wanted = [d.strip().lower() for d in args.datasets.split(",") if d.strip()]
    for name in wanted:
        print(f"… {name}", file=sys.stderr, flush=True)
        try:
            if name == "hotpotqa":
                datasets[name] = await run_hotpot(args.limit, reader, args.cpu_watts)
            elif name == "fever":
                datasets[name] = await run_fever(args.limit, reader, args.cpu_watts)
            else:
                print(f"jeu de données inconnu ignoré : {name}", file=sys.stderr)
        except LLMError as exc:
            print(f"{name} : lecteur LLM en échec ({exc}), relance hors ligne", file=sys.stderr)
            datasets[name] = await (run_hotpot if name == "hotpotqa" else run_fever)(args.limit, None, args.cpu_watts)
            status = f"en échec ({exc}) : mesures hors ligne"
    return {
        "version": __version__,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "seed": SEED,
        "host": host_info(),
        "llm": {"enabled": reader is not None, "status": status},
        "datasets": datasets,
        "process": {"peak_rss_mb": peak_rss_mb(), "vram_mb": 0},
        "systems": SYSTEM_LABELS,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m aiotech.bench", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", default="hotpotqa,fever")
    parser.add_argument("--limit", type=int, default=None, help="nombre maximal de questions par jeu")
    parser.add_argument("--llm", action="store_true", help="ajoute le lecteur LLM (payant, clé requise)")
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--cpu-watts", type=float, default=DEFAULT_CPU_WATTS,
                        help="puissance par cœur occupé pour l'estimation d'énergie sans RAPL")
    parser.add_argument("--out", type=Path, default=Path("bench-results"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s : %(message)s")
    report = asyncio.run(main_async(args))
    text = markdown(report)
    print(text)
    for note in write_outputs(args.out, report, text):
        print(note, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
