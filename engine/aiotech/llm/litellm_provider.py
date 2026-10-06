"""
Tous les autres LLM via LiteLLM : OpenAI, Mistral, Gemini, Groq, modèles locaux Ollama...

Le modèle se nomme à la manière LiteLLM (« openai/gpt-4.1-mini », « mistral/mistral-large-latest »,
« ollama/llama3.1 »). Les sorties typées sont demandées en JSON, puis validées par Pydantic :
une réponse non conforme lève LLMError et le pipeline se replie.
"""
from __future__ import annotations

import importlib
import json
import re
from collections.abc import AsyncIterator, Sequence
from typing import Any

from pydantic import BaseModel, ValidationError

from aiotech.llm.base import ChatMessage, Completion, Effort, LLMError, StreamDelta
from aiotech.llm.pricing import cost_usd

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def parse_json_model(text: str, schema: type[BaseModel]) -> BaseModel:
    match = _JSON_BLOCK.search(text)
    if match is None:
        raise LLMError("réponse sans objet JSON")
    try:
        return schema.model_validate(json.loads(match.group(0)))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise LLMError("JSON non conforme au schéma attendu") from exc


class LiteLLMProvider:
    name = "litellm"

    def __init__(self, model: str, api_base: str | None = None, timeout: float = 90.0) -> None:
        try:
            self._litellm: Any = importlib.import_module("litellm")
        except ImportError as exc:
            raise LLMError("LiteLLM n'est pas installé : pip install 'aiotech-search[providers]'") from exc
        self.model = model
        self.api_base = api_base or None
        self.timeout = timeout

    def _payload(self, system: str, messages: Sequence[ChatMessage], schema: type[BaseModel] | None) -> list[dict[str, str]]:
        instructions = system
        if schema is not None:
            instructions += (
                "\n\nRéponds uniquement par un objet JSON valide conforme à ce schéma JSON, sans texte autour :\n"
                + json.dumps(schema.model_json_schema(), ensure_ascii=False)
            )
        return [{"role": "system", "content": instructions}, *({"role": m.role, "content": m.content} for m in messages)]

    def _usage(self, response: Any) -> tuple[int, int, float | None]:
        usage = getattr(response, "usage", None)
        tokens_in = int(getattr(usage, "prompt_tokens", 0) or 0)
        tokens_out = int(getattr(usage, "completion_tokens", 0) or 0)
        try:
            cost: float | None = float(self._litellm.completion_cost(completion_response=response))
        except Exception:
            cost = cost_usd(self.model, tokens_in, tokens_out)
        return tokens_in, tokens_out, cost

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
            response = await self._litellm.acompletion(
                model=self.model, messages=self._payload(system, messages, schema), max_tokens=max_tokens,
                api_base=self.api_base, timeout=self.timeout,
            )
        except Exception as exc:
            raise LLMError(f"LiteLLM : {exc.__class__.__name__}") from exc
        choice = response.choices[0]
        if getattr(choice, "finish_reason", "") == "content_filter":
            raise LLMError("réponse filtrée par le fournisseur")
        text = str(choice.message.content or "")
        parsed = parse_json_model(text, schema) if schema is not None else None
        tokens_in, tokens_out, cost = self._usage(response)
        return Completion(text=text, parsed=parsed, tokens_in=tokens_in, tokens_out=tokens_out, cost_usd=cost,
                          model=self.model, provider=self.name)

    async def stream(
        self,
        *,
        system: str,
        messages: Sequence[ChatMessage],
        max_tokens: int,
        effort: Effort,
    ) -> AsyncIterator[StreamDelta | Completion]:
        try:
            response = await self._litellm.acompletion(
                model=self.model, messages=self._payload(system, messages, None), max_tokens=max_tokens,
                api_base=self.api_base, timeout=self.timeout, stream=True, stream_options={"include_usage": True},
            )
            parts: list[str] = []
            tokens_in = tokens_out = 0
            async for chunk in response:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    parts.append(delta)
                    yield StreamDelta(delta)
                usage = getattr(chunk, "usage", None)
                if usage is not None:
                    tokens_in = int(getattr(usage, "prompt_tokens", 0) or 0)
                    tokens_out = int(getattr(usage, "completion_tokens", 0) or 0)
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"LiteLLM : {exc.__class__.__name__}") from exc
        yield Completion(text="".join(parts), parsed=None, tokens_in=tokens_in, tokens_out=tokens_out,
                         cost_usd=cost_usd(self.model, tokens_in, tokens_out), model=self.model, provider=self.name)
