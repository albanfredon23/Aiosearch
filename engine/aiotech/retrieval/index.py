"""
Recherche hybride : BM25 (lexical exact) + vecteurs lexicaux hachés (critère `reach` de l'ARG),
fusionnés par rangs réciproques (Reciprocal Rank Fusion, k = 60).

La fusion par rangs évite de mélanger des échelles de scores incomparables : un passage
bien classé par les deux méthodes passe devant un passage excellent pour une seule.
Un passage trouvé par les seuls vecteurs (aucun mot commun avec la requête) doit atteindre
reach >= 0,3 : en dessous, la ressemblance ne tient qu'à des fragments de mots.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from aiotech.core.arg import reach_many
from aiotech.core.embeddings import HashingEmbedder, Matrix
from aiotech.core.text import content_words
from aiotech.models import Passage, ScoredPassage
from aiotech.retrieval.bm25 import BM25Index

RRF_K = 60.0


@dataclass
class PassageIndex:
    embedder: HashingEmbedder = field(default_factory=lambda: HashingEmbedder(bigram_weight=0.0))
    min_reach: float = 0.12
    vector_only_reach: float = 0.3
    passages: list[Passage] = field(default_factory=list)
    _bm25: BM25Index = field(default_factory=BM25Index)
    _matrix: Matrix = field(default_factory=lambda: np.zeros((0, 1), dtype=np.float32))

    def build(self, passages: Sequence[Passage]) -> None:
        self.passages = list(passages)
        texts = [p.text for p in self.passages]
        self._bm25.build(texts)
        self._matrix = self.embedder.embed(texts) if texts else np.zeros((0, self.embedder.dim), dtype=np.float32)

    def search(self, query: str, top_k: int) -> list[ScoredPassage]:
        if not self.passages or top_k <= 0:
            return []
        meaningful = content_words(query)
        q_vec = self.embedder.embed_one(" ".join(meaningful) if meaningful else query)
        reaches = reach_many(q_vec, self._matrix)
        bm25 = self._bm25.scores(query)

        pool = top_k * 4
        bm25_ranked = sorted(bm25.items(), key=lambda item: item[1], reverse=True)[:pool]
        vector_ranked = sorted(
            ((i, r) for i, r in enumerate(reaches) if r >= self.min_reach),
            key=lambda item: item[1], reverse=True,
        )[:pool]
        bm25_rank = {i: rank for rank, (i, _) in enumerate(bm25_ranked, start=1)}
        vector_rank = {i: rank for rank, (i, _) in enumerate(vector_ranked, start=1)}

        fused: list[ScoredPassage] = []
        for i in set(bm25_rank) | set(vector_rank):
            if i not in bm25_rank and reaches[i] < self.vector_only_reach:
                continue
            score = 0.0
            if i in bm25_rank:
                score += 1.0 / (RRF_K + bm25_rank[i])
            if i in vector_rank:
                score += 1.0 / (RRF_K + vector_rank[i])
            fused.append(
                ScoredPassage(
                    passage=self.passages[i],
                    score=round(score * RRF_K, 6),
                    bm25_rank=bm25_rank.get(i),
                    vector_rank=vector_rank.get(i),
                )
            )
        fused.sort(key=lambda sp: (sp.score, -(sp.bm25_rank or 10**6)), reverse=True)
        return fused[:top_k]
