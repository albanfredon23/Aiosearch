"""
Systèmes comparés sur les mêmes documents :

    rag              RAG classique : BM25 Okapi top-k sur les passages ; la « réponse » est la
                     phrase la mieux classée par BM25, indexée avec le titre de son document
                     (aucune vérification, aucune abstention).
    aiotech          AIOTECH Search tel que livré (porte de calcul automatique) : recherche hybride,
                     graphe, SCG/TAP, réponses vérifiées ; peut s'abstenir.
    aiotech-deep     même pipeline forcé au palier approfondi (second saut de recherche).
"""
from __future__ import annotations

from aiotech.core.adaptive import Depth
from aiotech.core.feedback import ClickFeedback
from aiotech.core.text import normalize, split_sentences
from aiotech.gateway.guardrails import Guardrails
from aiotech.models import Document, Passage, SearchResult
from aiotech.pipeline import SearchEngine, SearchOptions
from aiotech.retrieval.bm25 import BM25Index
from aiotech.retrieval.corpus import CorpusStore
from aiotech.storage.kv import MemoryKV


def build_store(documents: list[Document]) -> CorpusStore:
    store = CorpusStore()
    store.add_many(documents, persist=False)
    store.search("index", 1)
    return store


class ClassicRag:
    name = "rag"

    def __init__(self, store: CorpusStore) -> None:
        self.passages: list[Passage] = list(store.passages.values())
        self.index = BM25Index()
        self.index.build([p.text for p in self.passages])
        self.sentences: list[tuple[Passage, str]] = []
        indexed: list[str] = []
        for p in self.passages:
            title = store.documents[p.document_id].title
            for sentence in split_sentences(p.text):
                if normalize(sentence.rstrip(". ")) == normalize(title):
                    continue
                self.sentences.append((p, sentence))
                indexed.append(sentence if sentence.startswith(title) else f"{title}. {sentence}")
        self.sentence_index = BM25Index()
        self.sentence_index.build(indexed)

    def retrieve(self, query: str, k: int) -> list[Passage]:
        return [self.passages[i] for i, _ in self.index.top(query, k)]

    def top_sentence(self, query: str) -> tuple[str | None, list[str]]:
        best = self.sentence_index.top(query, 1)
        if not best:
            return None, []
        passage, sentence = self.sentences[best[0][0]]
        return sentence, [passage.document_id]


class Aiotech:
    def __init__(self, store: CorpusStore, depth: Depth = "auto") -> None:
        self.name = "aiotech" if depth == "auto" else f"aiotech-{depth}"
        self.depth: Depth = depth
        kv = MemoryKV()
        self.engine = SearchEngine(corpus=store, feedback=ClickFeedback(kv), cache=kv,
                                   guardrails=Guardrails(max_chars=4000))

    def retrieve(self, query: str, k: int) -> list[Passage]:
        passages, _ = self.engine.evidence(query, self.depth, top_k=k)
        return [sp.passage for sp in passages]

    async def search(self, query: str) -> SearchResult:
        return await self.engine.search(query, SearchOptions(depth=self.depth, use_web=False, use_llm=False))

    @staticmethod
    def top_answer(result: SearchResult) -> tuple[str | None, list[str]]:
        if not result.possibilities:
            return None, []
        top = result.possibilities[0]
        return top.answer, [s.id for s in top.sources]
