"""
Routeur LLM : chaîne de repli entre fournisseurs et disjoncteur par fournisseur.

Le premier fournisseur disponible répond ; un échec passe au suivant. Après
`failure_threshold` échecs consécutifs, un fournisseur est mis à l'écart pendant
`cooldown_seconds` (disjoncteur ouvert), puis retenté une fois (semi-ouvert).
Un refus de sécurité n'ouvre pas le disjoncteur : le fournisseur fonctionne.
"""
from __future__ import annotations

import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field

from pydantic import BaseModel

from aiotech.llm.base import ChatMessage, Completion, Effort, LLMError, LLMProvider, LLMRefusal, StreamDelta


@dataclass
class CircuitBreaker:
    failure_threshold: int = 3
    cooldown_seconds: float = 60.0
    failures: int = 0
    opened_at: float | None = None

    def available(self, now: float | None = None) -> bool:
        if self.opened_at is None:
            return True
        return (time.monotonic() if now is None else now) - self.opened_at >= self.cooldown_seconds

    def success(self) -> None:
        self.failures = 0
        self.opened_at = None

    def failure(self, now: float | None = None) -> None:
        self.failures += 1
        if self.failures >= self.failure_threshold:
            self.opened_at = time.monotonic() if now is None else now

    @property
    def state(self) -> str:
        if self.opened_at is None:
            return "fermé"
        return "semi-ouvert" if self.available() else "ouvert"


@dataclass
class LLMRouter:
    providers: Sequence[LLMProvider]
    breakers: dict[str, CircuitBreaker] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for provider in self.providers:
            self.breakers.setdefault(self._key(provider), CircuitBreaker())

    @staticmethod
    def _key(provider: LLMProvider) -> str:
        return f"{provider.name}:{provider.model}"

    @property
    def available(self) -> bool:
        return bool(self.providers)

    @property
    def primary_model(self) -> str | None:
        return self.providers[0].model if self.providers else None

    def status(self) -> list[dict[str, str | int]]:
        return [
            {"provider": p.name, "model": p.model, "circuit": self.breakers[self._key(p)].state,
             "failures": self.breakers[self._key(p)].failures}
            for p in self.providers
        ]

    async def complete(
        self,
        *,
        system: str,
        messages: Sequence[ChatMessage],
        max_tokens: int,
        effort: Effort,
        schema: type[BaseModel] | None = None,
    ) -> Completion:
        errors: list[str] = []
        for provider in self.providers:
            breaker = self.breakers[self._key(provider)]
            if not breaker.available():
                errors.append(f"{provider.model} : disjoncteur ouvert")
                continue
            try:
                result = await provider.complete(system=system, messages=messages, max_tokens=max_tokens,
                                                 effort=effort, schema=schema)
            except LLMRefusal as exc:
                breaker.success()
                errors.append(f"{provider.model} : {exc}")
                continue
            except LLMError as exc:
                breaker.failure()
                errors.append(f"{provider.model} : {exc}")
                continue
            breaker.success()
            return result
        raise LLMError("aucun LLM disponible (" + " ; ".join(errors) + ")" if errors else "aucun LLM configuré")

    async def stream(
        self,
        *,
        system: str,
        messages: Sequence[ChatMessage],
        max_tokens: int,
        effort: Effort,
    ) -> AsyncIterator[StreamDelta | Completion]:
        errors: list[str] = []
        for provider in self.providers:
            breaker = self.breakers[self._key(provider)]
            if not breaker.available():
                continue
            emitted = False
            try:
                async for item in provider.stream(system=system, messages=messages, max_tokens=max_tokens,
                                                  effort=effort):
                    emitted = True
                    yield item
            except LLMError as exc:
                if not isinstance(exc, LLMRefusal):
                    breaker.failure()
                if emitted:
                    raise
                errors.append(f"{provider.model} : {exc}")
                continue
            breaker.success()
            return
        raise LLMError("aucun LLM disponible (" + " ; ".join(errors) + ")" if errors else "aucun LLM configuré")
