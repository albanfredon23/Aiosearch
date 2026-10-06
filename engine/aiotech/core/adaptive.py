"""
AdaptiveComputeGate : le calcul suit la difficulté de la question (frugalité, Green AI).

Une question simple n'ouvre que deux trajectoires, cherche peu de passages et n'appelle
aucun LLM pour planifier ou extraire ; une question à contraintes multiples ou à
plusieurs sauts reçoit plus de trajectoires, un second saut de recherche et un effort
de raisonnement plus élevé.

Complexité c dans [0, 1] :
    0.30 · min(1, mots porteurs / 20)
  + 0.25 · min(1, contraintes / 3)
  + 0.25 si recommandation ou comparaison, 0.10 si factuelle, 0.05 si définition
  + 0.10 si au moins deux entités nommées
  + 0.10 si la question enchaîne deux faits (« le réalisateur du film qui... »)
Paliers : c < 0.30 léger, c < 0.60 standard, sinon approfondi.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from aiotech.core.text import content_words
from aiotech.models import ComputeBudget, IntentPlan

Tier = Literal["light", "standard", "deep"]
Depth = Literal["auto", "light", "standard", "deep"]

_MULTI_HOP_RE = re.compile(
    r"\b(?:dont|duquel|de laquelle|celui qui|celle qui|whose|of the \w+ (?:that|who|which)|"
    r"the \w+ (?:who|that|which) \w+ the|qui a (?:réalisé|écrit|fondé|dirigé))\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TierSpec:
    max_interpretations: int
    top_k: int
    context_tokens: int
    hops: int
    effort: Literal["low", "medium", "high"]
    llm_planner: bool
    llm_extraction: bool


TIERS: dict[Tier, TierSpec] = {
    "light": TierSpec(2, 5, 1200, 1, "low", False, False),
    "standard": TierSpec(3, 8, 2400, 1, "medium", True, True),
    "deep": TierSpec(4, 10, 4000, 2, "high", True, True),
}


def complexity(plan: IntentPlan) -> float:
    score = 0.30 * min(1.0, len(content_words(plan.query)) / 20.0)
    score += 0.25 * min(1.0, len(plan.constraints) / 3.0)
    score += {"recommendation": 0.25, "comparison": 0.25, "factual": 0.10, "definition": 0.05}[plan.question_type]
    if len(plan.entities) >= 2:
        score += 0.10
    if _MULTI_HOP_RE.search(plan.query):
        score += 0.10
    return round(min(1.0, score), 3)


def tier_for(score: float) -> Tier:
    if score < 0.30:
        return "light"
    if score < 0.60:
        return "standard"
    return "deep"


class AdaptiveComputeGate:
    def __init__(self, max_context_tokens: int = 4000, llm_available: bool = False) -> None:
        self.max_context_tokens = max_context_tokens
        self.llm_available = llm_available

    def budget(self, plan: IntentPlan, depth: Depth = "auto", use_llm: bool = True) -> ComputeBudget:
        score = complexity(plan)
        tier: Tier = tier_for(score) if depth == "auto" else depth
        spec = TIERS[tier]
        llm = self.llm_available and use_llm
        return ComputeBudget(
            tier=tier,
            complexity=score,
            max_interpretations=spec.max_interpretations,
            top_k=spec.top_k,
            context_tokens=min(spec.context_tokens, self.max_context_tokens),
            use_llm_planner=llm and spec.llm_planner,
            use_llm_extraction=llm and spec.llm_extraction,
            use_llm_synthesis=llm,
            effort=spec.effort,
            hops=spec.hops,
        )
