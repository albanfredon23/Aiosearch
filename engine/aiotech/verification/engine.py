"""
Verification Engine : un statut explicite pour chaque affirmation de la réponse.

    [FAIT]         citation retrouvée mot pour mot dans la source ET appui des sources
                   (OU bruité des fiabilités) >= seuil ET aucune contradiction ouverte ;
    [INFÉRENCE]    conclusion dérivée uniquement de prémisses [FAIT] ou [INFÉRENCE]
                   (respect d'une contrainte, d'une exigence) ;
    [INCERTAIN]    sources en désaccord sans vainqueur net, ou prémisse incertaine ;
    [NON VÉRIFIÉ]  appui insuffisant (source unique peu fiable) ou citation introuvable.

La confiance d'une inférence est la t-norme de Gödel de celles de ses prémisses :
une conclusion ne vaut jamais plus que sa prémisse la plus faible.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from aiotech.core.arg import godel_tnorm
from aiotech.core.text import normalize
from aiotech.core.units import format_quantity
from aiotech.graph.claims import ATTRIBUTE_LABELS
from aiotech.graph.knowledge_graph import KnowledgeGraph
from aiotech.models import Claim, Passage, Source, Status, ValueGroup, VerifiedClaim

_SPACES = re.compile(r"\s+")


def _flat(text: str) -> str:
    return _SPACES.sub(" ", normalize(text)).strip()


def quote_in_source(quote: str, source_text: str) -> bool:
    """Citation présente telle quelle (casse, accents et espaces normalisés)."""
    q = _flat(quote)
    return bool(q) and q in _flat(source_text)


def attribute_label(attribute: str) -> str:
    return ATTRIBUTE_LABELS.get(attribute, attribute)


def group_value(group: ValueGroup) -> str:
    return format_quantity(group.quantity) if group.quantity is not None and group.quantity.unit else group.value_text


def _source_names(sources: Sequence[Source], limit: int = 3) -> str:
    names = [s.title for s in sources[:limit]]
    extra = len(sources) - limit
    return ", ".join(names) + (f" et {extra} autre(s)" if extra > 0 else "")


class VerificationEngine:
    def __init__(self, fact_threshold: float = 0.6) -> None:
        if not 0.0 < fact_threshold <= 1.0:
            raise ValueError("le seuil de [FAIT] doit être dans ]0, 1]")
        self.fact_threshold = fact_threshold

    def _quotes_found(self, claims: Sequence[Claim], passages: Mapping[str, Passage]) -> bool:
        for claim in claims:
            passage = passages.get(claim.passage_id)
            if passage is None or not quote_in_source(claim.quote, passage.text):
                return False
        return True

    def verify_attribute(
        self, graph: KnowledgeGraph, entity_key: str, attribute: str, passages: Mapping[str, Passage]
    ) -> VerifiedClaim | None:
        resolution = graph.resolve(entity_key, attribute)
        if not resolution.groups:
            return None
        label = attribute_label(attribute)
        if resolution.conflict and not resolution.resolved:
            values = " ou ".join(group_value(g) for g in resolution.groups[:3])
            detail = " ; ".join(f"{group_value(g)} : appui {g.support:.2f}" for g in resolution.groups[:3])
            claim_ids = tuple(cid for g in resolution.groups for cid in g.claim_ids)
            return VerifiedClaim(
                text=f"{label} : {values}",
                status=Status.INCERTAIN,
                reason=f"Sources en désaccord sans vainqueur net ({detail})",
                sources=self._sources(graph, claim_ids),
                claim_ids=claim_ids,
                confidence=round(max(0.0, resolution.groups[0].support - resolution.groups[1].support), 4),
            )
        winner = resolution.winner
        assert winner is not None
        claims = [graph.claims[cid] for cid in winner.claim_ids]
        sources = self._sources(graph, winner.claim_ids)
        text = f"{label} : {group_value(winner)}"
        if not self._quotes_found(claims, passages):
            return VerifiedClaim(
                text=text, status=Status.NON_VERIFIE, reason="Citation introuvable dans la source",
                sources=sources, claim_ids=winner.claim_ids, confidence=0.0,
            )
        if winner.support < self.fact_threshold:
            return VerifiedClaim(
                text=text, status=Status.NON_VERIFIE,
                reason=f"Appui insuffisant ({winner.support:.2f} < {self.fact_threshold:.2f}) : {_source_names(sources)}",
                sources=sources, claim_ids=winner.claim_ids, confidence=winner.support,
            )
        reason = f"{len(sources)} source(s), appui {winner.support:.2f} : {_source_names(sources)}"
        if resolution.conflict:
            losers = ", ".join(f"{group_value(g)} (appui {g.support:.2f})" for g in resolution.groups[1:3])
            reason += f" ; valeur concurrente écartée : {losers}"
        return VerifiedClaim(
            text=text, status=Status.FAIT, reason=reason, sources=sources,
            claim_ids=winner.claim_ids, confidence=winner.support,
        )

    def verify_statement(
        self,
        text: str,
        quote: str,
        passage: Passage,
        corroborating: Sequence[Source] = (),
        disputed: bool = False,
    ) -> VerifiedClaim:
        """Phrase extraite d'une source : FAIT si citée exactement et assez appuyée."""
        sources = (passage.source, *[s for s in corroborating if s.id != passage.source.id])
        remaining = 1.0
        for s in sources:
            remaining *= 1.0 - s.reliability
        support = round(1.0 - remaining, 4)
        if not quote_in_source(quote, passage.text):
            return VerifiedClaim(text=text, status=Status.NON_VERIFIE, reason="Citation introuvable dans la source",
                                 sources=sources, confidence=0.0)
        if disputed:
            return VerifiedClaim(text=text, status=Status.INCERTAIN,
                                 reason="Une autre source affirme une valeur différente", sources=sources,
                                 confidence=round(support / 2, 4))
        if support < self.fact_threshold:
            return VerifiedClaim(
                text=text, status=Status.NON_VERIFIE,
                reason=f"Appui insuffisant ({support:.2f} < {self.fact_threshold:.2f}) : {_source_names(sources)}",
                sources=sources, confidence=support,
            )
        return VerifiedClaim(text=text, status=Status.FAIT,
                             reason=f"Cité mot pour mot, appui {support:.2f} : {_source_names(sources)}",
                             sources=sources, confidence=support)

    def verify_claim_set(self, text: str, claims: Sequence[Claim], passages: Mapping[str, Passage]) -> VerifiedClaim:
        """Affirmation portée par plusieurs extraits concordants (par exemple une exigence documentée)."""
        distinct: dict[str, Source] = {}
        for claim in claims:
            distinct.setdefault(claim.source.id, claim.source)
        sources = tuple(sorted(distinct.values(), key=lambda s: s.reliability, reverse=True))
        remaining = 1.0
        for s in sources:
            remaining *= 1.0 - s.reliability
        support = round(1.0 - remaining, 4)
        claim_ids = tuple(c.id for c in claims)
        if not claims or not self._quotes_found(claims, passages):
            return VerifiedClaim(text=text, status=Status.NON_VERIFIE, reason="Citation introuvable dans la source",
                                 sources=sources, claim_ids=claim_ids, confidence=0.0)
        if support < self.fact_threshold:
            return VerifiedClaim(
                text=text, status=Status.NON_VERIFIE,
                reason=f"Appui insuffisant ({support:.2f} < {self.fact_threshold:.2f}) : {_source_names(sources)}",
                sources=sources, claim_ids=claim_ids, confidence=support,
            )
        return VerifiedClaim(text=text, status=Status.FAIT,
                             reason=f"{len(sources)} source(s), appui {support:.2f} : {_source_names(sources)}",
                             sources=sources, claim_ids=claim_ids, confidence=support)

    @staticmethod
    def infer(text: str, premises: Sequence[VerifiedClaim], rule: str) -> VerifiedClaim:
        """Conclusion dérivée : son statut et sa confiance suivent la prémisse la plus faible."""
        statuses = {p.status for p in premises}
        confidence = round(godel_tnorm(p.confidence for p in premises), 4) if premises else 0.0
        sources = tuple({s.id: s for p in premises for s in p.sources}.values())
        claim_ids = tuple(dict.fromkeys(cid for p in premises for cid in p.claim_ids))
        if not premises or Status.NON_VERIFIE in statuses:
            status, reason = Status.NON_VERIFIE, f"{rule} ; une prémisse n'est pas vérifiée"
        elif Status.INCERTAIN in statuses:
            status, reason = Status.INCERTAIN, f"{rule} ; une prémisse est incertaine"
        else:
            status, reason = Status.INFERENCE, f"{rule} ; déduit de prémisses vérifiées"
        return VerifiedClaim(text=text, status=status, reason=reason, sources=sources, claim_ids=claim_ids,
                             confidence=confidence)

    @staticmethod
    def _sources(graph: KnowledgeGraph, claim_ids: Sequence[str]) -> tuple[Source, ...]:
        seen: dict[str, Source] = {}
        for cid in claim_ids:
            source = graph.claims[cid].source
            seen.setdefault(source.id, source)
        return tuple(sorted(seen.values(), key=lambda s: s.reliability, reverse=True))
