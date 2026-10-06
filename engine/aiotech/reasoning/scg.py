"""
SCG : prémisses d'un candidat et porte de Gödel.

Chaque candidat (un produit, une valeur, une phrase de source) est jugé sur des
prémisses à degré de vérité dans [0, 1] :
    pertinence    rang hybride du meilleur passage qui le mentionne ;
    facette       le candidat relève-t-il de l'interprétation de la trajectoire ;
    appui         OU bruité des fiabilités de ses sources ;
    contrainte    contraintes de la requête (« prix ≤ 500 € ») et exigences documentées
                  (« AIOTECH 44 nécessite au moins 16 Go de RAM ») ;
    cohérence     absence de contradiction ouverte dans le graphe.

    A(candidat) = T_G(p_1, ..., p_n) = min_i p_i

Une prémisse fausse (degré 0, par exemple une contrainte violée) rejette la trajectoire
du candidat ; une valeur inconnue vaut 1/2 (logique de Kleene : ni vrai ni faux).
Une exigence tirée d'un document n'est sûre qu'à hauteur de la fiabilité c de ce
document, d'où l'implication de Kleene-Dienes :  degré = max(1 − c, s).
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from aiotech.core.arg import godel_tnorm
from aiotech.core.text import has_phrase
from aiotech.core.units import COMPARATOR_LABELS, format_quantity, satisfies
from aiotech.graph.entities import entity_key
from aiotech.graph.knowledge_graph import KnowledgeGraph, noisy_or
from aiotech.intent.constraints import constraint_label
from aiotech.models import Claim, Constraint, Premise, Quantity

UNKNOWN = 0.5
REJECTION_FLOOR = 0.15
SIBLING_FACET = 0.1

Outcome = Literal["satisfied", "violated", "unknown", "uncertain"]


def kleene_dienes(confidence: float, satisfaction: float) -> float:
    """Implication floue (c → s) = max(1 − c, s)."""
    return max(1.0 - confidence, satisfaction)


@dataclass(frozen=True)
class ConstraintCheck:
    constraint: Constraint
    satisfaction: float
    outcome: Outcome
    observed: tuple[Quantity, ...]

    def describe(self) -> str:
        observed = " ou ".join(format_quantity(q) for q in self.observed) or "valeur inconnue"
        mark = {"satisfied": "respectée", "violated": "violée", "unknown": "non vérifiable",
                "uncertain": "incertaine"}[self.outcome]
        return f"{self.constraint.label} : {observed}, {mark}"


def _attributes_for(graph: KnowledgeGraph, key: str, constraint: Constraint) -> list[str]:
    if not constraint.attribute.startswith("unit:"):
        return [constraint.attribute]
    unit = constraint.attribute.removeprefix("unit:")
    return sorted({c.attribute for c in graph.claims_for(key) if c.quantity is not None and c.quantity.unit == unit})


def check_constraint(graph: KnowledgeGraph, key: str, constraint: Constraint) -> ConstraintCheck:
    observed: list[Quantity] = []
    outcomes: list[bool] = []
    for attribute in _attributes_for(graph, key, constraint):
        resolution = graph.resolve(key, attribute)
        groups = [resolution.winner] if resolution.winner is not None else list(resolution.groups)
        for group in groups:
            if group is None or group.quantity is None:
                continue
            verdict = satisfies(group.quantity, constraint.comparator, constraint.quantity)
            if verdict is None:
                continue
            observed.append(group.quantity)
            outcomes.append(verdict)
    if not outcomes:
        return ConstraintCheck(constraint, UNKNOWN, "unknown", ())
    if all(outcomes):
        return ConstraintCheck(constraint, 1.0, "satisfied", tuple(observed))
    if not any(outcomes):
        return ConstraintCheck(constraint, 0.0, "violated", tuple(observed))
    return ConstraintCheck(constraint, UNKNOWN, "uncertain", tuple(observed))


@dataclass(frozen=True)
class Requirement:
    """Exigence documentée qui s'applique aux candidats (« X nécessite au moins 16 Go »)."""

    constraint: Constraint
    confidence: float
    claim_ids: tuple[str, ...]


def requirements_for(graph: KnowledgeGraph, holders: Iterable[str]) -> list[Requirement]:
    """Exigences des entités nommées dans la requête, fusionnées par (attribut, comparateur, valeur)."""
    merged: dict[tuple[str, str, float, str], list[Claim]] = {}
    for holder in holders:
        for key in graph.find_entities(entity_key(holder)):
            for claim in graph.claims_for(key, kinds=("requirement",)):
                if claim.quantity is None or (claim.comparator == "eq" and claim.attribute in {"value", "capacity"}):
                    continue
                ident = (claim.attribute, claim.comparator, claim.quantity.value, claim.quantity.unit)
                merged.setdefault(ident, []).append(claim)
    out: list[Requirement] = []
    for (attribute, _, _, _), claims in merged.items():
        head = claims[0]
        assert head.quantity is not None
        holder_label = graph.entity_label(head.subject_key)
        distinct = {c.source.id: c.source.reliability for c in claims}
        constraint = Constraint(
            attribute=attribute,
            comparator=head.comparator,
            quantity=head.quantity,
            label=f"exigence {holder_label} : {constraint_label(attribute, head.comparator, head.quantity)}",
            origin="document",
            holder=holder_label,
            claim_id=head.id,
        )
        out.append(Requirement(constraint, round(noisy_or(distinct.values()), 4), tuple(c.id for c in claims)))
    out.sort(key=lambda r: (r.constraint.attribute, COMPARATOR_LABELS[r.constraint.comparator]))
    return out


def facet_degree(text: str, own_terms: Sequence[str], sibling_terms: Sequence[str]) -> float | None:
    """1 si le texte relève de la facette, 0,1 s'il relève d'une facette voisine, 1/2 sinon."""
    if not own_terms:
        return None
    if any(has_phrase(text, t) for t in own_terms):
        return 1.0
    if any(has_phrase(text, t) for t in sibling_terms if t not in own_terms):
        return SIBLING_FACET
    return UNKNOWN


@dataclass(frozen=True)
class Gate:
    admissibility: float
    rejected: bool
    reason: str | None


def godel_gate(premises: Sequence[Premise], floor: float = REJECTION_FLOOR) -> Gate:
    """A = min des prémisses ; rejet si A = 0 ou A < plancher, motivé par la prémisse la plus faible."""
    if not premises:
        return Gate(0.0, True, "Aucune prémisse évaluable")
    admissibility = round(godel_tnorm(p.degree for p in premises), 4)
    weakest = min(premises, key=lambda p: p.degree)
    if admissibility <= 0.0 or admissibility < floor:
        return Gate(admissibility, True, f"Prémisse fausse : {weakest.label} (degré {weakest.degree:.2f})")
    return Gate(admissibility, False, None)
