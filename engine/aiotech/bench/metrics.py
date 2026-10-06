"""
Mesures du benchmark : normalisation officielle HotpotQA (EM, F1), rappels, MRR, intervalles
de confiance à 95 % par bootstrap apparié (mêmes questions rééchantillonnées pour les deux systèmes).
"""
from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

BOOTSTRAP_SAMPLES = 2000


def normalize_answer(text: str) -> str:
    """Normalisation du script officiel HotpotQA : minuscules, ponctuation, articles, espaces."""
    lowered = text.lower()
    no_punct = "".join(ch for ch in lowered if ch not in set(string.punctuation))
    no_articles = re.sub(r"\b(a|an|the)\b", " ", no_punct)
    return " ".join(no_articles.split())


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def f1_score(prediction: str, gold: str) -> float:
    pred, ref = normalize_answer(prediction), normalize_answer(gold)
    if pred in {"yes", "no", "noanswer"} or ref in {"yes", "no", "noanswer"}:
        return float(pred == ref)
    pred_tokens, ref_tokens = pred.split(), ref.split()
    common = Counter(pred_tokens) & Counter(ref_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(pred_tokens), overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def contains_answer(text: str | None, gold: str) -> float:
    if not text:
        return 0.0
    needle = normalize_answer(gold)
    return float(bool(needle) and f" {needle} " in f" {normalize_answer(text)} ")


def recall_at(ranked: Sequence[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    return len(set(ranked[:k]) & gold) / len(gold)


def all_at(ranked: Sequence[str], gold: set[str], k: int) -> float:
    return float(bool(gold) and gold <= set(ranked[:k]))


def reciprocal_rank(ranked: Sequence[str], gold: set[str], k: int = 10) -> float:
    for position, item in enumerate(ranked[:k], start=1):
        if item in gold:
            return 1.0 / position
    return 0.0


def percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else 0.0


@dataclass(frozen=True)
class Estimate:
    mean: float
    low: float
    high: float

    def as_dict(self) -> dict[str, float]:
        return {"mean": round(self.mean, 4), "low": round(self.low, 4), "high": round(self.high, 4)}


def bootstrap(values: Sequence[float], seed: int) -> Estimate:
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        return Estimate(0.0, 0.0, 0.0)
    rng = np.random.default_rng(seed)
    means = data[rng.integers(0, data.size, size=(BOOTSTRAP_SAMPLES, data.size))].mean(axis=1)
    return Estimate(float(data.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def paired_difference(a: Sequence[float], b: Sequence[float], seed: int) -> Estimate:
    """Différence moyenne a − b sur les mêmes questions, IC 95 % par bootstrap apparié."""
    if len(a) != len(b):
        raise ValueError("échantillons appariés de tailles différentes")
    return bootstrap([x - y for x, y in zip(a, b, strict=True)], seed)
