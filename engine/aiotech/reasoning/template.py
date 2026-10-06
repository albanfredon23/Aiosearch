"""
Réponses gabarit : rédigées uniquement à partir des affirmations vérifiées, sans LLM.

Aucune valeur n'est inventée : chaque élément de la phrase vient d'une affirmation du
graphe, avec son statut. C'est la réponse par défaut hors ligne et le repli quand la
rédaction par LLM échoue au contrôle de chaîne ou au contrôle des nombres.
"""
from __future__ import annotations

from collections.abc import Sequence

from aiotech.core.text import normalize
from aiotech.models import Status, VerifiedClaim

_DERIVED_PREFIXES = ("Respecte ", "Ne respecte pas ", "Respect incertain de ")


def _is_derived(claim: VerifiedClaim) -> bool:
    return claim.text.startswith(_DERIVED_PREFIXES) or claim.text.endswith(": non vérifiable")


_GENERIC_LABELS = frozenset({"valeur", "capacité", "énoncé"})


def _value(claim: VerifiedClaim) -> str:
    label, _, value = claim.text.partition(" : ")
    if not value:
        return claim.text
    if label in _GENERIC_LABELS or normalize(label) in normalize(value):
        return value
    return f"{label} {value}"


def entity_answer(label: str, claims: Sequence[VerifiedClaim]) -> str:
    facts = [c for c in claims if not _is_derived(c) and not c.text.startswith("type : ")]
    kind = next((c.text.removeprefix("type : ") for c in claims if c.text.startswith("type : ")), None)
    head = f"{label} ({kind})" if kind else label
    parts = [head]
    if facts:
        parts[0] += " : " + ", ".join(_value(c) for c in facts[:4])
    respected = [c.text.removeprefix("Respecte ") for c in claims
                 if c.text.startswith("Respecte ") and c.status in (Status.INFERENCE, Status.FAIT)]
    doubtful = [c.text for c in claims if _is_derived(c) and c.status in (Status.INCERTAIN, Status.NON_VERIFIE)]
    sentences = [parts[0] + "."]
    if respected:
        sentences.append("Respecte " + " ; ".join(respected) + ".")
    if doubtful:
        sentences.append("À confirmer : " + " ; ".join(doubtful) + ".")
    return " ".join(sentences)


def value_answer(label: str, claims: Sequence[VerifiedClaim]) -> str:
    if not claims:
        return label + "."
    claim = claims[0]
    names = ", ".join(s.title for s in claim.sources[:2])
    return f"{label} ({names})." if names else f"{label}."
