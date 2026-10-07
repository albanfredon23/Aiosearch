"""
Index BM25 (Okapi) sur les mots porteurs de sens racinisés, avec index inversé :
le score d'une requête ne parcourt que les passages qui partagent au moins un terme.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

from aiotech.core.text import content_words


@dataclass
class BM25Index:
    k1: float = 1.5
    b: float = 0.75
    _postings: dict[str, list[tuple[int, int]]] = field(default_factory=lambda: defaultdict(list))
    _lengths: list[int] = field(default_factory=list)
    _avg_length: float = 0.0

    def build(self, texts: Sequence[str]) -> None:
        self._postings = defaultdict(list)
        self._lengths = []
        for doc_index, text in enumerate(texts):
            terms = content_words(text)
            self._lengths.append(len(terms))
            for term, tf in Counter(terms).items():
                self._postings[term].append((doc_index, tf))
        self._avg_length = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0

    @property
    def size(self) -> int:
        return len(self._lengths)

    def scores(self, query: str) -> dict[int, float]:
        n = self.size
        if n == 0:
            return {}
        result: dict[int, float] = defaultdict(float)
        for term in sorted(set(content_words(query))):  # ordre fixe : sommes identiques d'une exécution à l'autre
            postings = self._postings.get(term)
            if not postings:
                continue
            df = len(postings)
            idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            for doc_index, tf in postings:
                length_norm = 1.0 - self.b + self.b * self._lengths[doc_index] / (self._avg_length or 1.0)
                result[doc_index] += idf * tf * (self.k1 + 1.0) / (tf + self.k1 * length_norm)
        return dict(result)

    def top(self, query: str, k: int) -> list[tuple[int, float]]:
        ranked = sorted(self.scores(query).items(), key=lambda item: item[1], reverse=True)
        return ranked[:k]
