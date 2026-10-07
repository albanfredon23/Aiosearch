"""
Réglage de la recherche HORS de l'échantillon de test (reproduit les choix de RetrievalParams et VECTOR_WEIGHT).

    python -m aiotech.bench.tune --hotpot hotpot_dev_distractor_v1.json --n 1000 --seed 7

Les questions sont tirées parmi les 7 105 questions du dev « distractor » qui ne sont PAS dans les 300 du
benchmark (même tirage random.Random(20261006) que prepare.py, exclu ici). Chaque réglage est évalué sur la
seule étape de recherche (SearchEngine.evidence), comparée à BM25 seul ; la sortie est un tableau Markdown.
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
from collections.abc import Sequence
from pathlib import Path

from aiotech.bench.metrics import all_at, recall_at
from aiotech.bench.prepare import HOTPOT_SAMPLE, HOTPOT_SHA256, SEED, sha256
from aiotech.bench.systems import ClassicRag
from aiotech.core.feedback import ClickFeedback
from aiotech.models import Document, Passage
from aiotech.pipeline import RetrievalParams, SearchEngine
from aiotech.retrieval.corpus import CorpusStore
from aiotech.storage.kv import MemoryKV

VECTOR_WEIGHTS = (1.0, 0.25, 0.1, 0.0)
SUBQUERY_WEIGHTS = (1.0, 0.5)
LINK_BOOSTS = (0.0, 0.5, 1.0)
METRICS = ("r@2", "r@5", "all@2", "all@5")

Item = tuple[str, frozenset[str], list[Document]]


def tuning_items(source: Path, n: int, seed: int) -> list[Item]:
    digest = sha256(source)
    if digest != HOTPOT_SHA256:
        raise SystemExit(f"{source} : empreinte {digest} différente du fichier officiel attendu")
    data = json.loads(source.read_text("utf-8"))
    test = set(random.Random(SEED).sample(range(len(data)), HOTPOT_SAMPLE))
    pool = [i for i in range(len(data)) if i not in test]
    items: list[Item] = []
    for index in random.Random(seed).sample(pool, n):
        row = data[index]
        documents = [Document(id=f"p{i}", title=title, text=f"{title}. {' '.join(s.strip() for s in sentences).strip()}",
                              reliability=0.7) for i, (title, sentences) in enumerate(row["context"])]
        items.append((row["question"], frozenset(title for title, _ in row["supporting_facts"]), documents))
    return items


def scores(ranked_titles: Sequence[str], gold: frozenset[str]) -> tuple[float, ...]:
    gold_set = set(gold)
    return (recall_at(ranked_titles, gold_set, 2), recall_at(ranked_titles, gold_set, 5),
            all_at(ranked_titles, gold_set, 2), all_at(ranked_titles, gold_set, 5))


def titles_of(passages: Sequence[Passage], store: CorpusStore) -> list[str]:
    seen: list[str] = []
    for p in passages:
        if p.document_id not in seen:
            seen.append(p.document_id)
    return [store.documents[d].title for d in seen]


def mean_row(rows: list[tuple[float, ...]]) -> list[float]:
    return [round(100 * sum(col) / len(rows), 1) for col in zip(*rows, strict=True)]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hotpot", type=Path, required=True)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    items = tuning_items(args.hotpot, args.n, args.seed)
    print(f"Réglage : {len(items)} questions HotpotQA hors test, random.Random({args.seed})\n")
    print("| Réglage | " + " | ".join(METRICS) + " |\n|---|" + "---|" * len(METRICS))
    for weight in VECTOR_WEIGHTS:
        stores = []
        for _, _, documents in items:
            store = CorpusStore(vector_weight=weight)
            store.add_many(documents, persist=False)
            stores.append(store)
        if weight == VECTOR_WEIGHTS[0]:
            rows = [scores(titles_of(ClassicRag(store).retrieve(q, 10), store), gold)
                    for (q, gold, _), store in zip(items, stores, strict=True)]
            print("| BM25 seul (RAG classique) | " + " | ".join(map(str, mean_row(rows))) + " |", flush=True)
        for sub, boost in itertools.product(SUBQUERY_WEIGHTS, LINK_BOOSTS):
            params = RetrievalParams(subquery_weight=sub, link_boost=boost)
            rows = []
            for (q, gold, _), store in zip(items, stores, strict=True):
                kv = MemoryKV()
                engine = SearchEngine(corpus=store, feedback=ClickFeedback(kv), cache=kv, retrieval=params)
                passages, _ = engine.evidence(q, top_k=10)
                rows.append(scores(titles_of([sp.passage for sp in passages], store), gold))
            label = f"vecteurs {weight} · entités {sub} · liens {boost}"
            print(f"| {label} | " + " | ".join(map(str, mean_row(rows))) + " |", flush=True)


if __name__ == "__main__":
    main()
