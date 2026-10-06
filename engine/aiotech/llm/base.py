"""
Contrat commun des fournisseurs LLM (Claude par défaut, tout autre modèle via LiteLLM).
"""
from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, TypeVar

from pydantic import BaseModel

Effort = Literal["low", "medium", "high"]
Role = Literal["user", "assistant"]
T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str


@dataclass(frozen=True)
class Completion:
    text: str
    parsed: BaseModel | None
    tokens_in: int
    tokens_out: int
    cost_usd: float | None
    model: str
    provider: str


@dataclass(frozen=True)
class StreamDelta:
    text: str


class LLMError(RuntimeError):
    """Échec d'appel (réseau, quota, réponse invalide) : le pipeline se replie sans LLM."""


class LLMRefusal(LLMError):
    def __init__(self, category: str | None) -> None:
        super().__init__(f"refus du modèle ({category or 'catégorie non précisée'})")
        self.category = category


class LLMProvider(Protocol):
    name: str
    model: str

    async def complete(
        self,
        *,
        system: str,
        messages: Sequence[ChatMessage],
        max_tokens: int,
        effort: Effort,
        schema: type[BaseModel] | None = None,
    ) -> Completion: ...

    def stream(
        self,
        *,
        system: str,
        messages: Sequence[ChatMessage],
        max_tokens: int,
        effort: Effort,
    ) -> AsyncIterator[StreamDelta | Completion]: ...
