"""
Coût d'un appel en dollars : tarifs publics par million de tokens (entrée, sortie).

Un modèle absent de la table renvoie None : le coût est alors affiché « inconnu »
plutôt qu'inventé (LiteLLM fournit son propre calcul quand il connaît le modèle).
"""
from __future__ import annotations

PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-fable-5-1": (10.0, 50.0),
    "claude-fable-5": (10.0, 50.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


def price_for(model: str) -> tuple[float, float] | None:
    for known in sorted(PRICES_PER_MTOK, key=len, reverse=True):
        if model == known or model.startswith(known + "-") or model.startswith(known + "@"):
            return PRICES_PER_MTOK[known]
    return None


def cost_usd(model: str, tokens_in: int, tokens_out: int) -> float | None:
    prices = price_for(model)
    if prices is None:
        return None
    return round(tokens_in * prices[0] / 1e6 + tokens_out * prices[1] / 1e6, 8)
