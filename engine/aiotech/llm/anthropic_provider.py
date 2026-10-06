"""
Fournisseur par défaut : Claude via le SDK officiel Anthropic.

    modèle             claude-opus-5-5 par défaut (AIOTECH_LLM_MODEL pour un autre) ;
    effort             output_config.effort, piloté par l'AdaptiveComputeGate ;
    sorties typées     beta.messages.parse(output_format=ModèlePydantic) ;
    refus              stop_reason == "refusal" levé en LLMRefusal (aucun texte partiel gardé) ;
                       repli côté serveur `fallbacks="default"` sur les modèles qui le permettent.
"""
from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

import anthropic
from anthropic import AsyncAnthropic
from anthropic.types.beta import BetaMessageParam
from pydantic import BaseModel

from aiotech.llm.base import ChatMessage, Completion, Effort, LLMError, LLMRefusal, StreamDelta
from aiotech.llm.pricing import cost_usd

_EFFORT_FAMILIES = (
    "claude-opus-5", "claude-sonnet-5", "claude-fable-5", "claude-mythos-5",
    "claude-opus-4-8", "claude-opus-4-7", "claude-opus-4-6", "claude-sonnet-4-6", "claude-opus-4-5",
)
_FALLBACK_FAMILIES = ("claude-opus-5", "claude-sonnet-5", "claude-fable-5")
FALLBACK_BETA = "server-side-fallback-2026-07-01"


def supports_effort(model: str) -> bool:
    return model.startswith(_EFFORT_FAMILIES)


def supports_server_fallback(model: str) -> bool:
    return model.startswith(_FALLBACK_FAMILIES)


class AnthropicProvider:
    name = "anthropic"

    def __init__(
        self,
        api_key: str,
        model: str = "claude-opus-5-5",
        base_url: str | None = None,
        timeout: float = 90.0,
        max_retries: int = 2,
        server_fallbacks: bool = True,
    ) -> None:
        self.model = model
        self.server_fallbacks = server_fallbacks and supports_server_fallback(model)
        self.client = AsyncAnthropic(api_key=api_key, base_url=base_url or None, timeout=timeout,
                                     max_retries=max_retries)

    def _options(self, effort: Effort) -> dict[str, Any]:
        options: dict[str, Any] = {}
        if supports_effort(self.model):
            options["output_config"] = {"effort": effort}
        if self.server_fallbacks:
            options["betas"] = [FALLBACK_BETA]
            options["fallbacks"] = "default"
        return options

    @staticmethod
    def _messages(messages: Sequence[ChatMessage]) -> list[BetaMessageParam]:
        return [{"role": m.role, "content": m.content} for m in messages]

    def _completion(self, response: Any, parsed: BaseModel | None) -> Completion:
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            raise LLMRefusal(getattr(details, "category", None))
        text = "".join(block.text for block in response.content if getattr(block, "type", "") == "text")
        usage = response.usage
        tokens_in = int(usage.input_tokens or 0) + int(getattr(usage, "cache_creation_input_tokens", 0) or 0) \
            + int(getattr(usage, "cache_read_input_tokens", 0) or 0)
        tokens_out = int(usage.output_tokens or 0)
        model = str(getattr(response, "model", self.model) or self.model)
        return Completion(text=text, parsed=parsed, tokens_in=tokens_in, tokens_out=tokens_out,
                          cost_usd=cost_usd(model, tokens_in, tokens_out), model=model, provider=self.name)

    async def complete(
        self,
        *,
        system: str,
        messages: Sequence[ChatMessage],
        max_tokens: int,
        effort: Effort,
        schema: type[BaseModel] | None = None,
    ) -> Completion:
        try:
            if schema is not None:
                parsed_response = await self.client.beta.messages.parse(
                    model=self.model, max_tokens=max_tokens, system=system, messages=self._messages(messages),
                    output_format=schema, **self._options(effort),
                )
                if parsed_response.stop_reason == "refusal":
                    return self._completion(parsed_response, None)
                parsed = parsed_response.parsed_output
                if parsed is None:
                    raise LLMError("sortie structurée absente ou invalide")
                return self._completion(parsed_response, parsed)
            response = await self.client.beta.messages.create(
                model=self.model, max_tokens=max_tokens, system=system, messages=self._messages(messages),
                **self._options(effort),
            )
            return self._completion(response, None)
        except anthropic.APIError as exc:
            raise LLMError(f"Anthropic : {exc.__class__.__name__}") from exc

    async def stream(
        self,
        *,
        system: str,
        messages: Sequence[ChatMessage],
        max_tokens: int,
        effort: Effort,
    ) -> AsyncIterator[StreamDelta | Completion]:
        try:
            async with self.client.beta.messages.stream(
                model=self.model, max_tokens=max_tokens, system=system, messages=self._messages(messages),
                **self._options(effort),
            ) as stream:
                async for text in stream.text_stream:
                    yield StreamDelta(text)
                final = await stream.get_final_message()
            yield self._completion(final, None)
        except anthropic.APIError as exc:
            raise LLMError(f"Anthropic : {exc.__class__.__name__}") from exc
