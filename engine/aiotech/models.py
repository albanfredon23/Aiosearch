"""
Modèles de données partagés par tout le pipeline AIOTECH Search (Pydantic, typage strict).
"""
from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Status(StrEnum):
    """Statut de vérification de chaque affirmation de la réponse."""

    FAIT = "FAIT"
    INFERENCE = "INFÉRENCE"
    INCERTAIN = "INCERTAIN"
    NON_VERIFIE = "NON VÉRIFIÉ"


Comparator = Literal["eq", "le", "lt", "ge", "gt"]
QuestionType = Literal["factual", "definition", "recommendation", "comparison"]
Origin = Literal["corpus", "web"]
ClaimKind = Literal["fact", "requirement", "type", "statement"]
Extractor = Literal["rules", "llm", "pipeline"]


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Source(Frozen):
    id: str
    title: str
    url: str | None = None
    origin: Origin = "corpus"
    reliability: float = Field(ge=0.0, le=1.0)


class Document(Frozen):
    id: str
    title: str
    text: str
    url: str | None = None
    reliability: float | None = Field(default=None, ge=0.0, le=1.0)
    origin: Origin = "corpus"
    metadata: dict[str, str] = Field(default_factory=dict)


class Passage(Frozen):
    id: str
    document_id: str
    text: str
    source: Source


class ScoredPassage(Frozen):
    passage: Passage
    score: float
    bm25_rank: int | None = None
    vector_rank: int | None = None


class Quantity(Frozen):
    """Valeur numérique dans une unité canonique (EUR, GB, day, % ...)."""

    value: float
    unit: str = ""


class Claim(Frozen):
    id: str
    subject: str
    subject_key: str
    attribute: str
    value_text: str
    quantity: Quantity | None = None
    comparator: Comparator = "eq"
    kind: ClaimKind = "fact"
    quote: str
    passage_id: str
    source: Source
    extractor: Extractor = "rules"


class Constraint(Frozen):
    attribute: str
    comparator: Comparator
    quantity: Quantity
    label: str
    origin: Literal["query", "document"] = "query"
    holder: str | None = None
    claim_id: str | None = None


class ValueGroup(Frozen):
    value_text: str
    quantity: Quantity | None
    claim_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    support: float


class Contradiction(Frozen):
    entity_key: str
    entity_label: str
    attribute: str
    groups: tuple[ValueGroup, ...]
    resolved: bool
    winner: str | None = None
    margin: float = 0.0


class Interpretation(Frozen):
    """Un neurone de l'interface : une lecture possible de la requête."""

    id: str
    label: str
    facet: str
    rationale: str
    subqueries: tuple[str, ...]
    facet_terms: tuple[str, ...] = ()
    plausibility: float = Field(default=0.5, ge=0.0, le=1.0)


class IntentPlan(Frozen):
    query: str
    objective: str
    question_type: QuestionType
    constraints: tuple[Constraint, ...]
    key_terms: tuple[str, ...]
    entities: tuple[str, ...]
    interpretations: tuple[Interpretation, ...]
    planner: Literal["rules", "llm"] = "rules"


class Premise(Frozen):
    label: str
    degree: float = Field(ge=0.0, le=1.0)
    kind: Literal["relevance", "facet", "evidence", "constraint", "consistency"]


class VerifiedClaim(Frozen):
    text: str
    status: Status
    reason: str
    sources: tuple[Source, ...] = ()
    claim_ids: tuple[str, ...] = ()
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class TapScore(Frozen):
    admissibility: float
    coherence: float
    support: float
    faithfulness: float
    contradiction_penalty: float
    quality: float
    reliability: float


class Candidate(Frozen):
    key: str
    label: str
    premises: tuple[Premise, ...]
    claims: tuple[VerifiedClaim, ...]
    score: TapScore
    rejected: bool
    rejection_reason: str | None = None


class Trajectory(Frozen):
    interpretation_id: str
    admissible: bool
    best: Candidate | None
    candidates: tuple[Candidate, ...]
    passages: int
    rejection_reason: str | None = None


class Possibility(Frozen):
    """Une des 2 ou 3 réponses finales vers lesquelles le nuage se contracte."""

    rank: int
    interpretation_id: str
    interpretation_label: str
    title: str
    answer: str
    reliability: float
    reliability_pct: int
    click_prior: float
    claims: tuple[VerifiedClaim, ...]
    sources: tuple[Source, ...]
    alternatives: tuple[str, ...] = ()
    synthesis: Literal["template", "llm"] = "template"


class ComputeBudget(Frozen):
    tier: Literal["light", "standard", "deep"]
    complexity: float
    max_interpretations: int
    top_k: int
    context_tokens: int
    use_llm_planner: bool
    use_llm_extraction: bool
    use_llm_synthesis: bool
    effort: Literal["low", "medium", "high"]
    hops: int = Field(default=1, ge=1, le=3)


class PiiFinding(Frozen):
    type: str
    count: int


class Usage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    llm_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float | None = 0.0

    def add(self, tokens_in: int, tokens_out: int, cost: float | None) -> None:
        self.llm_calls += 1
        self.tokens_in += tokens_in
        self.tokens_out += tokens_out
        if cost is None or self.cost_usd is None:
            self.cost_usd = None
        else:
            self.cost_usd = round(self.cost_usd + cost, 8)


class SearchEvent(Frozen):
    type: Literal[
        "start", "blocked", "interpretations", "retrieval", "graph", "trajectory",
        "possibilities", "warning", "done", "error",
    ]
    data: dict[str, Any]


class SearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    search_id: str
    query: str
    blocked: bool = False
    block_reason: str | None = None
    pii: list[PiiFinding] = Field(default_factory=list)
    compute: ComputeBudget | None = None
    plan: IntentPlan | None = None
    trajectories: list[Trajectory] = Field(default_factory=list)
    possibilities: list[Possibility] = Field(default_factory=list)
    contradictions: list[Contradiction] = Field(default_factory=list)
    graph: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    metrics: dict[str, float | int | bool | None] = Field(default_factory=dict)
