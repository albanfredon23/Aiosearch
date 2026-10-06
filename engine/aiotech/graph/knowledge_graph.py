"""
Graphe de connaissances d'une recherche : entités, affirmations, sources, confiance.

La projection en graphe rend les contradictions mécaniques : pour une même entité et un
même attribut, deux valeurs qui ne sont pas égales (à la tolérance d'unité près) forment
une contradiction X ≠ Y. Chaque groupe de valeurs reçoit un appui en « OU bruité » sur
ses sources distinctes :

    appui(G) = 1 - Π_{s ∈ sources(G)} (1 - fiabilité(s))

Une contradiction est résolue quand l'appui du premier groupe dépasse celui du second
d'au moins `resolution_margin` ; sinon elle reste ouverte et les affirmations concernées
seront marquées [INCERTAIN].
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from aiotech.core.text import content_words, normalize
from aiotech.core.units import same_value
from aiotech.graph.entities import EntityResolver, same_entity
from aiotech.models import Claim, Contradiction, Source, ValueGroup

_NON_CONFLICTING_ATTRIBUTES = frozenset({"type", "statement"})
_SINGLE_VALUED_TEXT = frozenset(
    {
        "capital", "date", "birth_date", "death_date", "birthplace", "founded", "founding_date", "release_date",
        "release_year", "author", "director", "founder", "ceo", "headquarters", "nationality", "country",
        "inventor", "creator", "composer", "spouse", "height", "duration", "deadline",
    }
)


def same_text_value(a: str, b: str) -> bool:
    """Deux valeurs textuelles concordent si l'une contient tous les mots de l'autre (« Nolan » ⊂ « Christopher Nolan »)."""
    if normalize(a) == normalize(b):
        return True
    wa, wb = set(content_words(a)), set(content_words(b))
    return bool(wa) and bool(wb) and (wa <= wb or wb <= wa)


def noisy_or(reliabilities: Iterable[float]) -> float:
    remaining = 1.0
    for r in reliabilities:
        remaining *= 1.0 - min(1.0, max(0.0, r))
    return 1.0 - remaining


@dataclass(frozen=True)
class Resolution:
    groups: tuple[ValueGroup, ...]
    winner: ValueGroup | None
    conflict: bool
    resolved: bool
    margin: float


@dataclass
class KnowledgeGraph:
    resolution_margin: float = 0.25
    value_tolerance: float = 0.02
    resolver: EntityResolver = field(default_factory=EntityResolver)
    claims: dict[str, Claim] = field(default_factory=dict)
    by_entity: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    sources: dict[str, Source] = field(default_factory=dict)

    def add_claims(self, claims: Iterable[Claim]) -> list[Claim]:
        """Ajoute des affirmations (sujets résolus vers leur entité canonique) ; renvoie celles ajoutées."""
        added: list[Claim] = []
        for claim in claims:
            if claim.id in self.claims or not claim.subject_key:
                continue
            canonical = self.resolver.resolve(claim.subject_key, claim.subject)
            stored = claim if canonical == claim.subject_key else claim.model_copy(update={"subject_key": canonical})
            self.claims[stored.id] = stored
            self.by_entity[canonical].append(stored.id)
            self.sources[stored.source.id] = stored.source
            added.append(stored)
        return added

    def entity_label(self, key: str) -> str:
        return self.resolver.label(key)

    def canonical_key(self, key: str) -> str:
        return self.resolver.canonical.get(key, key)

    def entity_keys(self) -> list[str]:
        return list(self.by_entity)

    def claims_for(self, entity_key: str, attribute: str | None = None,
                   kinds: tuple[str, ...] = ("fact",)) -> list[Claim]:
        key = self.canonical_key(entity_key)
        out = [self.claims[cid] for cid in self.by_entity.get(key, [])]
        return [c for c in out if c.kind in kinds and (attribute is None or c.attribute == attribute)]

    def find_entities(self, key: str) -> list[str]:
        """Entités canoniques qui désignent `key` (même règle de fusion que la résolution)."""
        return [k for k in self.by_entity if same_entity(k, key)]

    def _groups(self, claims: list[Claim]) -> list[ValueGroup]:
        clusters: list[list[Claim]] = []
        for claim in claims:
            for cluster in clusters:
                head = cluster[0]
                if claim.quantity is not None and head.quantity is not None:
                    matched = same_value(claim.quantity, head.quantity, self.value_tolerance)
                else:
                    matched = (
                        claim.quantity is None and head.quantity is None
                        and same_text_value(claim.value_text, head.value_text)
                    )
                if matched:
                    cluster.append(claim)
                    break
            else:
                clusters.append([claim])
        groups: list[ValueGroup] = []
        for cluster in clusters:
            distinct: dict[str, float] = {}
            for c in cluster:
                distinct[c.source.id] = max(distinct.get(c.source.id, 0.0), c.source.reliability)
            groups.append(
                ValueGroup(
                    value_text=cluster[0].value_text,
                    quantity=cluster[0].quantity,
                    claim_ids=tuple(c.id for c in cluster),
                    source_ids=tuple(distinct),
                    support=round(noisy_or(distinct.values()), 4),
                )
            )
        groups.sort(key=lambda g: (g.support, len(g.source_ids)), reverse=True)
        return groups

    def resolve(self, entity_key: str, attribute: str) -> Resolution:
        """Valeur(s) d'un attribut : groupe gagnant, conflit éventuel, résolu ou non."""
        exact = [c for c in self.claims_for(entity_key, attribute) if c.comparator == "eq"]
        groups = tuple(self._groups(exact))
        if not groups:
            return Resolution((), None, False, False, 0.0)
        quantified = any(g.quantity is not None for g in groups)
        if len(groups) == 1 or attribute in _NON_CONFLICTING_ATTRIBUTES or (
            not quantified and attribute not in _SINGLE_VALUED_TEXT
        ):
            return Resolution(groups, groups[0], False, True, groups[0].support)
        margin = round(groups[0].support - groups[1].support, 4)
        resolved = margin >= self.resolution_margin
        return Resolution(groups, groups[0] if resolved else None, True, resolved, margin)

    def contradictions(self) -> list[Contradiction]:
        found: list[Contradiction] = []
        for key, claim_ids in self.by_entity.items():
            attributes = {
                self.claims[cid].attribute for cid in claim_ids
                if self.claims[cid].kind == "fact" and self.claims[cid].attribute not in _NON_CONFLICTING_ATTRIBUTES
            }
            for attribute in sorted(attributes):
                res = self.resolve(key, attribute)
                if not res.conflict:
                    continue
                found.append(
                    Contradiction(
                        entity_key=key,
                        entity_label=self.entity_label(key),
                        attribute=attribute,
                        groups=res.groups,
                        resolved=res.resolved,
                        winner=res.winner.value_text if res.winner else None,
                        margin=res.margin,
                    )
                )
        return found

    def export(self) -> dict[str, Any]:
        """Vue sérialisable : entités, affirmations, sources, contradictions."""
        return {
            "entities": [
                {"key": key, "label": self.entity_label(key), "claims": len(ids)}
                for key, ids in self.by_entity.items()
            ],
            "claims": [
                {
                    "id": c.id, "entity": c.subject_key, "attribute": c.attribute, "value": c.value_text,
                    "comparator": c.comparator, "kind": c.kind, "source": c.source.id, "extractor": c.extractor,
                }
                for c in self.claims.values()
            ],
            "sources": [s.model_dump() for s in self.sources.values()],
            "contradictions": [c.model_dump() for c in self.contradictions()],
        }
