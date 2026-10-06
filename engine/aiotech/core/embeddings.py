"""
Embeddings lexicaux déterministes (HashingEmbedder, repris du cœur aiotech v45).

Mots racinisés, bigrammes de mots et trigrammes de caractères, répartis par hachage sur
`dim` dimensions positives, pondération TF sous-linéaire. Aucun modèle, aucun appel réseau,
coût nul : deux textes qui partagent du vocabulaire ont des vecteurs proches.

Les vecteurs ne sont pas projetés sur la sphère unité : l'ARG raisonne en distance
euclidienne avec des critères à unité explicite (voir core/arg.py).
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from itertools import pairwise
from typing import Protocol

import numpy as np
import numpy.typing as npt

from aiotech.core.text import stem, words

Vector = npt.NDArray[np.float32]
Matrix = npt.NDArray[np.float32]


class Embedder(Protocol):
    dim: int

    def embed_one(self, text: str) -> Vector: ...

    def embed(self, texts: Sequence[str]) -> Matrix: ...


def _bucket(feature: str, dim: int) -> int:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % dim


class HashingEmbedder:
    def __init__(
        self,
        dim: int = 2048,
        char_ngrams: int = 3,
        word_weight: float = 1.0,
        bigram_weight: float = 0.7,
        char_weight: float = 0.25,
    ) -> None:
        if dim <= 0:
            raise ValueError("dim doit être strictement positif")
        self.dim = dim
        self.char_ngrams = char_ngrams
        self.word_weight = word_weight
        self.bigram_weight = bigram_weight
        self.char_weight = char_weight

    def _features(self, text: str) -> dict[str, float]:
        feats: dict[str, float] = {}
        tokens = words(text)
        for w in tokens:
            key = "w:" + stem(w)
            feats[key] = feats.get(key, 0.0) + self.word_weight
        if self.bigram_weight > 0.0:
            for a, b in pairwise(tokens):
                key = "b:" + a + "_" + b
                feats[key] = feats.get(key, 0.0) + self.bigram_weight
        n = self.char_ngrams
        for w in tokens:
            padded = f" {w} "
            for i in range(max(0, len(padded) - n + 1)):
                key = "c:" + padded[i : i + n]
                feats[key] = feats.get(key, 0.0) + self.char_weight
        return feats

    def embed_one(self, text: str) -> Vector:
        vec = np.zeros(self.dim, dtype=np.float32)
        for feature, tf in self._features(text).items():
            # TF sous-linéaire : 1 + log(tf) atténue les répétitions.
            vec[_bucket(feature, self.dim)] += 1.0 + math.log(tf) if tf >= 1.0 else tf
        return vec

    def embed(self, texts: Sequence[str]) -> Matrix:
        if len(texts) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self.embed_one(t) for t in texts])


def weighted_jaccard(a: Vector, b: Vector) -> float:
    """Recouvrement de deux vecteurs positifs : somme des min / somme des max, dans [0, 1]."""
    num = float(np.minimum(a, b).sum())
    den = float(np.maximum(a, b).sum())
    return num / den if den > 0.0 else 0.0
