"""
Lecteur LLM commun aux deux systèmes (mêmes consignes, même modèle, même budget de preuves) :
seule la sélection et la vérification des preuves diffèrent entre RAG classique et AIOTECH.
"""
from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel

from aiotech.core.text import estimate_tokens
from aiotech.llm.base import ChatMessage, Completion, LLMError
from aiotech.llm.router import LLMRouter

EVIDENCE_TOKENS = 700

ANSWER_SYSTEM = (
    "Answer the question using only the numbered evidence. Reply with the shortest exact answer span "
    "(a name, a date, a number, a few words) or yes/no. If the evidence does not contain the answer, reply unknown."
)
LABEL_SYSTEM = (
    "Decide from the numbered evidence only whether the claim is SUPPORTS (the evidence entails it), "
    "REFUTES (the evidence contradicts it) or NOT ENOUGH INFO."
)

Label = Literal["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]


class ShortAnswer(BaseModel):
    answer: str


class Verdict(BaseModel):
    label: Label


def evidence_block(texts: Sequence[str], budget: int = EVIDENCE_TOKENS) -> str:
    lines: list[str] = []
    used = 0
    for text in texts:
        cost = estimate_tokens(text)
        if used + cost > budget:
            break
        lines.append(f"[{len(lines) + 1}] {text}")
        used += cost
    return "\n".join(lines) or "(no evidence)"


class Reader:
    def __init__(self, router: LLMRouter) -> None:
        self.router = router

    async def _ask(self, system: str, prompt: str, schema: type[BaseModel]) -> tuple[BaseModel, Completion, float]:
        started = time.perf_counter()
        completion = await self.router.complete(system=system, messages=[ChatMessage("user", prompt)], max_tokens=200,
                                                effort="low", schema=schema)
        latency = (time.perf_counter() - started) * 1000.0
        if not isinstance(completion.parsed, schema):
            raise LLMError("réponse LLM hors schéma")
        return completion.parsed, completion, latency

    async def answer(self, question: str, evidence: Sequence[str]) -> tuple[str, Completion, float]:
        prompt = f"Evidence:\n{evidence_block(evidence)}\n\nQuestion: {question}"
        parsed, completion, latency = await self._ask(ANSWER_SYSTEM, prompt, ShortAnswer)
        assert isinstance(parsed, ShortAnswer)
        return parsed.answer.strip(), completion, latency

    async def label(self, claim: str, evidence: Sequence[str]) -> tuple[Label, Completion, float]:
        prompt = f"Evidence:\n{evidence_block(evidence)}\n\nClaim: {claim}"
        parsed, completion, latency = await self._ask(LABEL_SYSTEM, prompt, Verdict)
        assert isinstance(parsed, Verdict)
        return parsed.label, completion, latency
