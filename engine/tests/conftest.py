from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

import pytest
from pydantic import BaseModel

from aiotech.core.feedback import ClickFeedback
from aiotech.llm.base import ChatMessage, Completion, Effort, StreamDelta
from aiotech.llm.router import LLMRouter
from aiotech.llm.tasks import LLMTasks
from aiotech.pipeline import SearchEngine
from aiotech.retrieval.corpus import CorpusStore
from aiotech.runtime import demo_documents
from aiotech.storage.kv import MemoryKV


class FakeProvider:
    """Fournisseur LLM factice : renvoie des sorties préparées par type de schéma et compte les appels."""

    name = "fake"

    def __init__(self, outputs: dict[type[BaseModel], BaseModel] | None = None, text: str = "Réponse.") -> None:
        self.model = "fake-model"
        self.outputs = outputs or {}
        self.text = text
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        *,
        system: str,
        messages: Sequence[ChatMessage],
        max_tokens: int,
        effort: Effort,
        schema: type[BaseModel] | None = None,
    ) -> Completion:
        self.calls.append({"system": system, "messages": list(messages), "schema": schema, "effort": effort})
        parsed = self.outputs.get(schema) if schema is not None else None
        return Completion(text=self.text, parsed=parsed, tokens_in=100, tokens_out=20, cost_usd=0.001,
                          model=self.model, provider=self.name)

    async def stream(
        self,
        *,
        system: str,
        messages: Sequence[ChatMessage],
        max_tokens: int,
        effort: Effort,
    ) -> AsyncIterator[StreamDelta | Completion]:
        self.calls.append({"system": system, "messages": list(messages), "schema": None, "effort": effort})
        for part in self.text.split(" "):
            yield StreamDelta(part + " ")
        yield Completion(text=self.text, parsed=None, tokens_in=50, tokens_out=10, cost_usd=0.0005,
                         model=self.model, provider=self.name)


@pytest.fixture
def demo_corpus() -> CorpusStore:
    corpus = CorpusStore()
    corpus.add_many(demo_documents(), persist=False)
    return corpus


@pytest.fixture
def kv() -> MemoryKV:
    return MemoryKV()


@pytest.fixture
def engine(demo_corpus: CorpusStore, kv: MemoryKV) -> SearchEngine:
    return SearchEngine(corpus=demo_corpus, feedback=ClickFeedback(kv), cache=kv)


def engine_with_llm(corpus: CorpusStore, provider: FakeProvider) -> SearchEngine:
    store = MemoryKV()
    return SearchEngine(corpus=corpus, feedback=ClickFeedback(store), cache=store,
                        llm=LLMTasks(LLMRouter([provider])))
