"""
Retour des clics : estimation en ligne du taux de choix de chaque interprétation.

Reprise de l'estimateur RLS (moindres carrés récursifs avec oubli) d'aiotech, réduit à
un modèle constant par facette : pour chaque affichage, y = 1 si la réponse a été
choisie, 0 si elle était montrée sans être choisie.

    k = P / (λ + P)
    p ← p + k (y − p)
    P ← (1 − k) P / λ

λ < 1 oublie les vieux clics (les usages changent). L'estimation ne sert qu'à départager
des réponses de fiabilité égale : la fiabilité reste le critère principal du classement.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from aiotech.core.text import normalize
from aiotech.storage.kv import KeyValueStore

_PREFIX = "feedback:"


@dataclass(frozen=True)
class RlsState:
    p: float = 0.5
    P: float = 1.0
    n: int = 0

    def update(self, y: float, forgetting: float) -> RlsState:
        gain = self.P / (forgetting + self.P)
        p = self.p + gain * (y - self.p)
        covariance = max(1e-4, min(1e3, (1.0 - gain) * self.P / forgetting))
        return RlsState(p=min(1.0, max(0.0, p)), P=covariance, n=self.n + 1)


def facet_key(facet: str) -> str:
    return normalize(facet).strip().replace(" ", "-")[:80] or "general"


class ClickFeedback:
    def __init__(self, store: KeyValueStore, forgetting: float = 0.98) -> None:
        if not 0.5 <= forgetting <= 1.0:
            raise ValueError("le facteur d'oubli doit être dans [0.5, 1]")
        self.store = store
        self.forgetting = forgetting

    async def state(self, facet: str) -> RlsState:
        raw = await self.store.get(_PREFIX + facet_key(facet))
        if raw is None:
            return RlsState()
        data = json.loads(raw)
        return RlsState(p=float(data["p"]), P=float(data["P"]), n=int(data["n"]))

    async def prior(self, facet: str) -> float:
        return (await self.state(facet)).p

    async def priors(self, facets: list[str]) -> dict[str, float]:
        return {facet: await self.prior(facet) for facet in dict.fromkeys(facets)}

    async def record(self, chosen: str, shown: list[str]) -> dict[str, float]:
        """Met à jour la facette choisie (y = 1) et les autres facettes montrées (y = 0)."""
        updated: dict[str, float] = {}
        for facet in dict.fromkeys([chosen, *shown]):
            y = 1.0 if facet == chosen else 0.0
            new = (await self.state(facet)).update(y, self.forgetting)
            await self.store.set(_PREFIX + facet_key(facet), json.dumps({"p": new.p, "P": new.P, "n": new.n}))
            updated[facet] = round(new.p, 4)
        return updated
