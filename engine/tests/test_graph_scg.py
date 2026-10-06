from __future__ import annotations

import pytest

from aiotech.graph.claims import RuleExtractor, query_attributes
from aiotech.graph.knowledge_graph import KnowledgeGraph, noisy_or, same_text_value
from aiotech.models import Constraint, Passage, Premise, Quantity, Source
from aiotech.reasoning.scg import (
    REJECTION_FLOOR,
    check_constraint,
    facet_degree,
    godel_gate,
    kleene_dienes,
    requirements_for,
)


def passage(pid: str, text: str, reliability: float) -> Passage:
    return Passage(id=pid, document_id=pid, text=text,
                   source=Source(id=pid, title=pid, reliability=reliability))


def graph_of(*passages: Passage) -> KnowledgeGraph:
    graph = KnowledgeGraph()
    extractor = RuleExtractor()
    for p in passages:
        claims, _ = extractor.extract(p)
        graph.add_claims(claims)
    return graph


def test_extractor_reads_subject_attribute_and_requirement() -> None:
    claims, focus = RuleExtractor().extract(passage(
        "fiche", "Le Nova Book 15 est un ordinateur portable de 15 pouces. Il coûte 479 € et embarque 16 Go de RAM.", 0.8))
    by_attribute = {c.attribute: c for c in claims}
    assert focus == "Nova Book 15"
    assert by_attribute["type"].value_text == "ordinateur portable"
    assert by_attribute["price"].quantity == Quantity(value=479.0, unit="EUR")
    assert by_attribute["ram"].quantity == Quantity(value=16.0, unit="GB")
    assert all(c.quote in "Le Nova Book 15 est un ordinateur portable de 15 pouces. Il coûte 479 € et embarque 16 Go de RAM."
               for c in claims)
    requirement, _ = RuleExtractor().extract(passage(
        "conf", "AIOTECH 44 nécessite au moins 16 Go de RAM.", 0.9))
    assert requirement[0].kind == "requirement" and requirement[0].comparator == "ge"


def test_query_attributes() -> None:
    assert query_attributes("Quel est le prix du Nova Book 15 ?") == {"price"}
    assert query_attributes("Qui a écrit ce livre ?") == set()


def test_contradiction_is_detected_and_resolved_by_reliability() -> None:
    graph = graph_of(
        passage("fiche", "Le Nova Book 15 coûte 479 €.", 0.8),
        passage("comparatif", "Le Nova Book 15 est affiché à 479 euros.", 0.7),
        passage("forum", "Le Nova Book 15 coûte 529 €.", 0.35),
    )
    contradictions = graph.contradictions()
    assert len(contradictions) == 1
    found = contradictions[0]
    assert found.attribute == "price" and found.resolved and found.winner is not None
    assert found.winner.startswith("479")
    assert found.groups[0].support == round(noisy_or([0.8, 0.7]), 4)


def test_close_contradiction_stays_open() -> None:
    graph = graph_of(
        passage("officiel", "L'Orion Mini PC coûte 429 €.", 0.6),
        passage("revendeur", "L'Orion Mini PC coûte 459 €.", 0.55),
    )
    found = graph.contradictions()
    assert len(found) == 1 and not found[0].resolved and found[0].winner is None
    key = graph.entity_keys()[0]
    budget = Constraint(attribute="price", comparator="le", quantity=Quantity(value=450.0, unit="EUR"), label="prix ≤ 450 €")
    check = check_constraint(graph, key, budget)
    assert check.outcome == "uncertain" and check.satisfaction == 0.5


def test_text_values_group_by_word_subset() -> None:
    assert same_text_value("Nolan", "Christopher Nolan")
    assert not same_text_value("Paris", "Lyon")


def test_constraint_check_outcomes() -> None:
    graph = graph_of(passage("fiche", "Le Zenit Air 15 coûte 549 €. Il embarque 16 Go de RAM.", 0.7))
    key = graph.entity_keys()[0]
    price = Constraint(attribute="price", comparator="le", quantity=Quantity(value=500.0, unit="EUR"), label="prix ≤ 500 €")
    ram = Constraint(attribute="ram", comparator="ge", quantity=Quantity(value=16.0, unit="GB"), label="RAM ≥ 16 Go")
    storage = Constraint(attribute="storage", comparator="ge", quantity=Quantity(value=512.0, unit="GB"), label="SSD ≥ 512 Go")
    assert check_constraint(graph, key, price).outcome == "violated"
    assert check_constraint(graph, key, price).satisfaction == 0.0
    assert check_constraint(graph, key, ram).outcome == "satisfied"
    assert check_constraint(graph, key, storage).outcome == "unknown"
    assert "violée" in check_constraint(graph, key, price).describe()


def test_documented_requirement_uses_kleene_dienes() -> None:
    graph = graph_of(
        passage("conf", "AIOTECH 44 nécessite au moins 16 Go de RAM.", 0.9),
        passage("aria", "L'Aria Book 14 coûte 399 €. Il embarque 8 Go de RAM.", 0.7),
    )
    requirements = requirements_for(graph, ["AIOTECH 44"])
    assert len(requirements) == 1
    req = requirements[0]
    assert req.confidence == 0.9 and req.constraint.origin == "document"
    aria = next(k for k in graph.entity_keys() if "aria" in k)
    check = check_constraint(graph, aria, req.constraint)
    assert check.outcome == "violated"
    assert kleene_dienes(req.confidence, check.satisfaction) == pytest.approx(0.1)
    assert kleene_dienes(0.3, 0.0) == pytest.approx(0.7)


def test_facet_degree() -> None:
    assert facet_degree("Nova Book 15, ordinateur portable", ["portable"], ["fixe"]) == 1.0
    assert facet_degree("Orion Mini PC, PC fixe", ["portable"], ["fixe"]) == 0.1
    assert facet_degree("Un ordinateur", ["portable"], ["fixe"]) == 0.5
    assert facet_degree("Un ordinateur", [], ["fixe"]) is None


def test_godel_gate_rejects_false_premise_and_low_admissibility() -> None:
    ok = godel_gate([Premise(label="pertinence", degree=0.8, kind="relevance"),
                     Premise(label="appui", degree=0.6, kind="evidence")])
    assert not ok.rejected and ok.admissibility == 0.6
    false = godel_gate([Premise(label="pertinence", degree=0.9, kind="relevance"),
                        Premise(label="prix ≤ 500 €", degree=0.0, kind="constraint")])
    assert false.rejected and false.reason is not None and "prix ≤ 500 €" in false.reason
    weak = godel_gate([Premise(label="facette", degree=REJECTION_FLOOR - 0.05, kind="facet")])
    assert weak.rejected
    assert godel_gate([]).rejected
