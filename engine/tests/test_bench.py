from __future__ import annotations

from pathlib import Path

import pytest

from aiotech.bench.__main__ import main, run_fever, run_hotpot
from aiotech.bench.datasets import load_fever, load_hotpot
from aiotech.bench.metrics import (
    all_at,
    bootstrap,
    contains_answer,
    exact_match,
    f1_score,
    normalize_answer,
    paired_difference,
    recall_at,
    reciprocal_rank,
)
from aiotech.bench.reader import Reader, ShortAnswer, Verdict, evidence_block
from aiotech.llm.router import LLMRouter
from tests.conftest import FakeProvider


def test_official_normalisation_and_scores() -> None:
    assert normalize_answer("The  Eiffel-Tower!") == "eiffeltower"
    assert exact_match("the Charmed", "Charmed") == 1.0
    assert f1_score("Christopher Nolan", "Nolan") == pytest.approx(2 / 3)
    assert f1_score("yes", "no") == 0.0
    assert contains_answer("She starred in Charmed from 2001.", "Charmed") == 1.0
    assert contains_answer("She starred in Charmedish.", "Charmed") == 0.0
    assert contains_answer(None, "Charmed") == 0.0


def test_ranking_metrics() -> None:
    ranked = ["a", "b", "c", "d"]
    assert recall_at(ranked, {"b", "z"}, 2) == 0.5
    assert all_at(ranked, {"a", "c"}, 2) == 0.0 and all_at(ranked, {"a", "c"}, 3) == 1.0
    assert reciprocal_rank(ranked, {"c"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(ranked, {"z"}) == 0.0


def test_bootstrap_is_deterministic_and_paired() -> None:
    values = [1.0, 0.0, 1.0, 1.0, 0.0] * 20
    first, second = bootstrap(values, 7), bootstrap(values, 7)
    assert first == second and first.low <= first.mean <= first.high
    assert paired_difference(values, values, 7).mean == 0.0
    with pytest.raises(ValueError):
        paired_difference([1.0], [1.0, 0.0], 7)


def test_frozen_samples() -> None:
    hotpot = load_hotpot()
    assert len(hotpot) == 300 and len({i.id for i in hotpot}) == 300
    assert all(len(i.paragraphs) >= 2 and i.supporting_titles for i in hotpot)
    fever = load_fever()
    assert len(fever.claims) == 355
    assert {c.label for c in fever.claims} == {"SUPPORTS", "REFUTES"}
    assert all(c.evidence_id in fever.evidence and c.evidence_id not in fever.synthetic for c in fever.claims)
    assert "-LRB-" not in " ".join(fever.evidence.values())


def test_evidence_budget() -> None:
    block = evidence_block(["court"] * 3 + ["long " * 2000])
    assert block.startswith("[1] court") and "[4]" not in block


async def test_offline_runs_without_errors() -> None:
    hotpot = await run_hotpot(4, None, 15.0)
    assert hotpot["n"] == 4 and hotpot["errors"] == {}
    assert {m["metric"] for m in hotpot["metrics"]} >= {"sp_recall@2", "answered"}
    assert hotpot["efficiency"]["aiotech"]["vram_mb"] == 0
    assert hotpot["efficiency"]["rag"]["cost_usd_per_query"] == 0.0
    fever = await run_fever(4, None, 15.0)
    assert fever["n"] == 4 and fever["errors"] == {}


async def test_llm_reader_path_counts_tokens_and_cost() -> None:
    provider = FakeProvider(outputs={ShortAnswer: ShortAnswer(answer="unknown"), Verdict: Verdict(label="SUPPORTS")})
    reader = Reader(LLMRouter([provider]))
    hotpot = await run_hotpot(2, reader, 15.0)
    assert "llm_f1" in {m["metric"] for m in hotpot["metrics"]}
    assert hotpot["efficiency"]["rag"]["llm_calls"] == 2
    assert hotpot["efficiency"]["aiotech"]["tokens_in_per_query"] == 100.0
    fever = await run_fever(2, reader, 15.0)
    accuracy = next(m for m in fever["metrics"] if m["metric"] == "llm_label_accuracy")
    assert set(accuracy["values"]) == {"rag", "aiotech"}


def test_cli_without_key_or_writable_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                            capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    blocker = tmp_path / "fichier"
    blocker.write_text("x")
    assert main(["--limit", "2", "--llm", "--out", str(blocker / "sous-dossier")]) == 0
    captured = capsys.readouterr()
    assert "# Benchmark AIOTECH Search" in captured.out
    assert "ANTHROPIC_API_KEY absente" in captured.out
    assert "résultats non écrits" in captured.err
    assert main(["--limit", "2", "--datasets", "fever", "--out", str(tmp_path / "ok")]) == 0
    assert (tmp_path / "ok" / "results.json").is_file()
