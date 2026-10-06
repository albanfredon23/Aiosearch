from __future__ import annotations

import numpy as np
import pytest

from aiotech.core.adaptive import AdaptiveComputeGate, complexity, parse_depth
from aiotech.core.arg import ArgConstraints, ReachabilityGate, godel_tnorm, reach
from aiotech.core.embeddings import HashingEmbedder
from aiotech.core.feedback import ClickFeedback
from aiotech.core.text import has_phrase, split_sentences
from aiotech.core.units import format_quantity, iter_quantities, parse_number, same_value, satisfies
from aiotech.intent.planner import RulePlanner
from aiotech.models import Quantity
from aiotech.storage.kv import MemoryKV


def test_godel_tnorm_is_min_and_empty_is_true() -> None:
    assert godel_tnorm([0.9, 0.4, 0.7]) == 0.4
    assert godel_tnorm([]) == 1.0
    assert godel_tnorm([1.0, 0.0]) == 0.0


def test_reach_bounds_and_containment() -> None:
    q = np.array([1.0, 1.0, 0.0], dtype=np.float32)
    assert reach(q, np.array([1.0, 1.0, 5.0], dtype=np.float32)) == pytest.approx(1.0)
    assert reach(q, np.zeros(3, dtype=np.float32)) == pytest.approx(0.0)
    assert 0.0 < reach(q, np.array([1.0, 0.0, 0.0], dtype=np.float32)) < 1.0


def test_admissibility_rejects_forbidden_terms() -> None:
    gate = ReachabilityGate()
    rows = gate.admissibility("délai de rétractation", ["Le délai de rétractation est de 14 jours.",
                                                         "Recette de la tarte aux pommes."],
                              ArgConstraints(forbidden_terms=("tarte",)))
    assert rows[0][3] > rows[1][3]
    assert rows[1][2] == 0.0 and rows[1][3] == 0.0


def test_verify_chain_flags_unreachable_step() -> None:
    gate = ReachabilityGate()
    context = ["Le délai de rétractation est de 14 jours pour un achat en ligne."]
    good = gate.verify_chain("Le délai de rétractation est de 14 jours.", context=context)
    bad = gate.verify_chain("Le délai de rétractation est de 14 jours. La garantie couvre les volcans martiens.",
                            context=context)
    assert good.admissible
    assert not bad.admissible and bad.weakest_step == 1


def test_embedder_is_deterministic() -> None:
    e = HashingEmbedder()
    assert np.array_equal(e.embed_one("ordinateur portable"), e.embed_one("ordinateur portable"))


@pytest.mark.parametrize(
    ("raw", "value"), [("1 299,99", 1299.99), ("1,299.50", 1299.5), ("4,5", 4.5), ("0,750", 0.75), ("16", 16.0)]
)
def test_parse_number(raw: str, value: float) -> None:
    assert parse_number(raw) == pytest.approx(value)


def test_units_are_canonical_and_comparable() -> None:
    found = [q for _, q in iter_quantities("16 Go de RAM, 1 To de SSD, 479 €, 2 semaines, de 2 to 5")]
    assert Quantity(value=16.0, unit="GB") in found
    assert Quantity(value=1000.0, unit="GB") in found
    assert Quantity(value=479.0, unit="EUR") in found
    assert Quantity(value=14.0, unit="day") in found
    assert not any(q.value == 2.0 and q.unit == "GB" for q in found)
    assert same_value(Quantity(value=479.0, unit="EUR"), Quantity(value=479.5, unit="EUR"))
    assert satisfies(Quantity(value=479.0, unit="EUR"), "le", Quantity(value=500.0, unit="EUR")) is True
    assert satisfies(Quantity(value=8.0, unit="GB"), "ge", Quantity(value=16.0, unit="GB")) is False
    assert satisfies(Quantity(value=8.0, unit="GB"), "ge", Quantity(value=16.0, unit="EUR")) is None
    assert format_quantity(Quantity(value=479.0, unit="EUR")) == "479 €"


def test_sentences_keep_decimals() -> None:
    assert split_sentences("Il coûte 4,5 €. Il pèse 1.5 kg.") == ["Il coûte 4,5 €.", "Il pèse 1.5 kg."]


def test_has_phrase_matches_whole_words_and_plurals() -> None:
    assert has_phrase("Quelles sont les exceptions ?", "exception")
    assert not has_phrase("faire tourner AIOTECH", "tour")
    assert has_phrase("Le droit ne s'applique pas aux denrées", "ne s'applique pas")


@pytest.mark.parametrize(
    ("query", "question_type"),
    [
        ("Quel ordinateur à 500 € serait le meilleur pour faire tourner AIOTECH 44 ?", "recommendation"),
        ("iPhone 15 vs Samsung Galaxy S24", "comparison"),
        ("Qu'est-ce que la photosynthèse ?", "definition"),
        ("Quel est le délai de rétractation pour un achat en ligne ?", "factual"),
    ],
)
def test_planner_routes_question_types(query: str, question_type: str) -> None:
    plan = RulePlanner().plan(query)
    assert plan.question_type == question_type
    assert [i.id for i in plan.interpretations] == [f"T{n}" for n in range(1, len(plan.interpretations) + 1)]
    plausibilities = [i.plausibility for i in plan.interpretations]
    assert plausibilities == sorted(plausibilities, reverse=True)


def test_planner_extracts_constraints_entities_and_facets() -> None:
    plan = RulePlanner().plan("Quel ordinateur à 500 € serait le meilleur pour faire tourner AIOTECH 44 ?")
    assert [c.label for c in plan.constraints] == ["prix ≤ 500 €"]
    assert plan.entities == ("AIOTECH 44",)
    assert {i.facet for i in plan.interpretations} == {"portable", "fixe", "occasion"}
    explicit = RulePlanner().plan("Quelles sont les exceptions au droit de rétractation ?")
    assert explicit.interpretations[0].facet == "exceptions"
    assert explicit.interpretations[0].plausibility == 0.9
    multi = RulePlanner().plan("PC avec 16 Go RAM et 512 Go SSD pour 600 euros maximum")
    assert [c.label for c in multi.constraints] == ["mémoire vive ≥ 16 Go", "stockage ≥ 512 Go", "prix ≤ 600 €"]


def test_compute_gate_scales_with_complexity() -> None:
    planner = RulePlanner()
    simple = planner.plan("Qu'est-ce que la photosynthèse ?")
    hard = planner.plan("Quel PC portable choisir pour moins de 450 euros avec 16 Go de RAM et 512 Go de SSD ?")
    assert complexity(simple) < complexity(hard)
    offline = AdaptiveComputeGate(llm_available=False)
    light, deep = offline.budget(simple), offline.budget(hard)
    assert light.tier == "light" and light.max_interpretations == 2
    assert deep.max_interpretations > light.max_interpretations
    assert not deep.use_llm_planner and not deep.use_llm_synthesis
    online = AdaptiveComputeGate(llm_available=True)
    assert online.budget(hard).use_llm_extraction
    assert not online.budget(simple).use_llm_planner
    assert not online.budget(hard, use_llm=False).use_llm_synthesis
    assert online.budget(simple, depth="deep").hops == 2
    assert parse_depth("DEEP") == "deep" and parse_depth("n'importe") == "auto"


async def test_click_feedback_moves_priors_within_bounds() -> None:
    feedback = ClickFeedback(MemoryKV())
    assert await feedback.prior("portable") == 0.5
    for _ in range(5):
        priors = await feedback.record("fixe", ["portable", "fixe", "occasion"])
    assert priors["fixe"] > 0.5 > priors["portable"]
    assert all(0.0 <= p <= 1.0 for p in priors.values())
