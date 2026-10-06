from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

import pytest
from pydantic import BaseModel

from aiotech.llm.base import ChatMessage, Completion, Effort, LLMError, LLMRefusal, StreamDelta
from aiotech.llm.pricing import cost_usd
from aiotech.llm.router import CircuitBreaker, LLMRouter
from aiotech.llm.tasks import (
    LLMAnswer,
    LLMAnswers,
    LLMClaim,
    LLMClaims,
    LLMInterpretation,
    LLMPlan,
    LLMTasks,
    normalize_attribute,
    numbers_supported,
    value_in_quote,
)
from aiotech.models import Passage, Source
from aiotech.pipeline import SearchEngine, SearchOptions
from aiotech.retrieval.corpus import CorpusStore
from tests.conftest import FakeProvider, engine_with_llm

PC_QUERY = "Quel ordinateur à 500 € serait le meilleur pour faire tourner AIOTECH 44 ?"


class FailingProvider:
    name = "failing"

    def __init__(self, error: LLMError) -> None:
        self.model = "failing-model"
        self.error = error
        self.calls = 0

    async def complete(self, *, system: str, messages: Sequence[ChatMessage], max_tokens: int, effort: Effort,
                       schema: type[BaseModel] | None = None) -> Completion:
        self.calls += 1
        raise self.error

    async def stream(self, *, system: str, messages: Sequence[ChatMessage], max_tokens: int,
                     effort: Effort) -> AsyncIterator[StreamDelta | Completion]:
        self.calls += 1
        raise self.error
        yield StreamDelta("")


async def test_router_falls_back_and_opens_breaker() -> None:
    failing = FailingProvider(LLMError("timeout"))
    backup = FakeProvider(text="ok")
    router = LLMRouter([failing, backup])
    for _ in range(3):
        result = await router.complete(system="s", messages=[ChatMessage("user", "q")], max_tokens=10, effort="low")
        assert result.provider == "fake"
    assert failing.calls == 3
    await router.complete(system="s", messages=[ChatMessage("user", "q")], max_tokens=10, effort="low")
    assert failing.calls == 3, "le disjoncteur ouvert ne doit plus solliciter le fournisseur en panne"
    states = {row["model"]: row["circuit"] for row in router.status()}
    assert states == {"failing-model": "ouvert", "fake-model": "fermé"}


async def test_refusal_does_not_open_breaker() -> None:
    refusing = FailingProvider(LLMRefusal("cyber"))
    router = LLMRouter([refusing])
    for _ in range(4):
        with pytest.raises(LLMError):
            await router.complete(system="s", messages=[ChatMessage("user", "q")], max_tokens=10, effort="low")
    assert refusing.calls == 4


async def test_stream_falls_back_before_first_token() -> None:
    router = LLMRouter([FailingProvider(LLMError("down")), FakeProvider(text="bonjour à tous")])
    parts = [item async for item in router.stream(system="s", messages=[ChatMessage("user", "q")], max_tokens=10,
                                                  effort="low")]
    assert "".join(p.text for p in parts if isinstance(p, StreamDelta)).strip() == "bonjour à tous"
    assert isinstance(parts[-1], Completion)


def test_circuit_breaker_half_opens_after_cooldown() -> None:
    breaker = CircuitBreaker(failure_threshold=2, cooldown_seconds=10.0)
    breaker.failure(now=100.0)
    breaker.failure(now=100.0)
    assert not breaker.available(now=105.0)
    assert breaker.available(now=111.0)
    breaker.success()
    assert breaker.failures == 0


def test_pricing() -> None:
    assert cost_usd("claude-opus-5-5", 1_000_000, 0) == pytest.approx(4.0)
    assert cost_usd("claude-haiku-4-5-20251001", 0, 1_000_000) == pytest.approx(5.0)
    assert cost_usd("modele-inconnu", 1000, 1000) is None


def test_guard_helpers() -> None:
    assert normalize_attribute("Prix") == "price"
    assert value_in_quote("479 €", "Le Nova Book 15 coûte 479 €.")
    assert not value_in_quote("399 €", "Le Nova Book 15 coûte 479 €.")
    assert numbers_supported("Il coûte 1 299 €.", ["Prix : 1299 €"])
    assert not numbers_supported("Il coûte 1 399 €.", ["Prix : 1299 €"])


async def test_llm_plan_replaces_rule_interpretations(demo_corpus: CorpusStore) -> None:
    plan = LLMPlan(objective="Choisir un PC", question_type="recommendation", interpretations=[
        LLMInterpretation(label="Portable léger", facet="Portable", rationale="mobilité",
                          subqueries=["ordinateur portable 16 Go RAM moins de 500 €"], facet_terms=["portable"],
                          plausibility=0.8),
        LLMInterpretation(label="Tour évolutive", facet="Fixe", rationale="puissance",
                          subqueries=["PC fixe tour moins de 500 €"], facet_terms=["fixe", "tour"], plausibility=1.4),
    ])
    provider = FakeProvider(outputs={LLMPlan: plan})
    engine = engine_with_llm(demo_corpus, provider)
    result = await engine.search(PC_QUERY, SearchOptions(depth="deep"))
    assert result.plan is not None and result.plan.planner == "llm"
    assert [i.label for i in result.plan.interpretations] == ["Portable léger", "Tour évolutive"]
    assert result.plan.interpretations[1].plausibility == 1.0
    assert [c.label for c in result.plan.constraints] == ["prix ≤ 500 €"]
    assert result.usage.llm_calls >= 1 and result.usage.tokens_in > 0
    assert provider.calls[0]["effort"] == "high"


def _passage(text: str) -> Passage:
    return Passage(id="p1", document_id="d1", text=text, source=Source(id="d1", title="Fiche", reliability=0.8))


async def test_llm_extraction_drops_unquoted_or_fabricated_claims() -> None:
    text = "Le Nova Book 15 coûte 479 € et embarque 16 Go de RAM."
    claims = LLMClaims(claims=[
        LLMClaim(passage_id="p1", subject="Nova Book 15", attribute="prix", value="479 €", comparator="eq",
                 kind="fact", quote="Le Nova Book 15 coûte 479 €"),
        LLMClaim(passage_id="p1", subject="Nova Book 15", attribute="prix", value="399 €", comparator="eq",
                 kind="fact", quote="Le Nova Book 15 coûte 479 €"),
        LLMClaim(passage_id="p1", subject="Nova Book 15", attribute="autonomie", value="12 heures", comparator="eq",
                 kind="fact", quote="Le Nova Book 15 tient 12 heures sur batterie"),
        LLMClaim(passage_id="p9", subject="Nova Book 15", attribute="ram", value="16 Go", comparator="eq",
                 kind="fact", quote="embarque 16 Go de RAM"),
    ])
    tasks = LLMTasks(LLMRouter([FakeProvider(outputs={LLMClaims: claims})]))
    extracted, completion = await tasks.extract("prix du Nova Book 15", [_passage(text)], 1000, "low")
    assert completion is not None
    assert [(c.attribute, c.value_text) for c in extracted] == [("price", "479 €")]
    assert extracted[0].extractor == "llm" and extracted[0].quantity is not None


async def test_llm_synthesis_is_rejected_when_unsupported(engine: SearchEngine) -> None:
    base = await engine.search(PC_QUERY)
    passages = dict(engine.corpus.passages)
    good = LLMAnswers(answers=[LLMAnswer(rank=1, answer="Le Nova Book 15 coûte 479 € et embarque 16 Go de RAM.")])
    tasks = LLMTasks(LLMRouter([FakeProvider(outputs={LLMAnswers: good})]))
    outcome, _ = await tasks.synthesize(PC_QUERY, base.possibilities, passages, "medium")
    assert outcome.possibilities[0].synthesis == "llm" and outcome.warnings == []
    assert outcome.possibilities[1].synthesis == "template"

    invented = LLMAnswers(answers=[LLMAnswer(rank=1, answer="Le Nova Book 15 coûte 399 € avec 32 Go de RAM.")])
    tasks = LLMTasks(LLMRouter([FakeProvider(outputs={LLMAnswers: invented})]))
    outcome, _ = await tasks.synthesize(PC_QUERY, base.possibilities, passages, "medium")
    assert outcome.possibilities[0].synthesis == "template"
    assert outcome.possibilities[0].answer == base.possibilities[0].answer
    assert any("nombre absent des preuves" in w for w in outcome.warnings)

    drifting = LLMAnswers(answers=[LLMAnswer(
        rank=1, answer="Le Nova Book 15 est recommandé. Les licornes martiennes adorent la photosynthèse quantique.")])
    tasks = LLMTasks(LLMRouter([FakeProvider(outputs={LLMAnswers: drifting})]))
    outcome, _ = await tasks.synthesize(PC_QUERY, base.possibilities, passages, "medium")
    assert outcome.possibilities[0].synthesis == "template"
    assert any("non atteignable" in w for w in outcome.warnings)
