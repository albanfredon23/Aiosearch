"""
Pipeline complet d'AIOTECH Search, diffusé étape par étape (SSE) :

    garde-fous      masquage RGPD, blocage des injections (0 token payé si bloqué)
    intention       type de question, contraintes, interprétations T1..Tn (neurones)
    budget          AdaptiveComputeGate : trajectoires, passages, sauts, effort, appels LLM
    recherche       par trajectoire : BM25 + vecteurs (corpus) et SearXNG (web), saut par les liens
                    de titre entre documents, 2e saut par entités pont si budget
    graphe          affirmations (règles, et LLM avec citation vérifiée), contradictions X ≠ Y
    raisonnement    SCG (porte de Gödel) puis TAP (fiabilité) par trajectoire
    vérification    [FAIT] / [INFÉRENCE] / [INCERTAIN] / [NON VÉRIFIÉ] pour chaque affirmation
    réponses        2 ou 3 possibilités classées par fiabilité, puis par clics à égalité
    rédaction       gabarit, ou LLM sous contrôle des nombres et de la chaîne ARG
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from aiotech.core.adaptive import AdaptiveComputeGate, Depth
from aiotech.core.arg import ReachabilityGate
from aiotech.core.feedback import ClickFeedback
from aiotech.core.text import content_words
from aiotech.gateway.guardrails import Guardrails
from aiotech.graph.knowledge_graph import KnowledgeGraph
from aiotech.intent.planner import RulePlanner, query_entities
from aiotech.llm.base import LLMError
from aiotech.llm.tasks import LLMTasks
from aiotech.models import (
    Claim,
    ComputeBudget,
    Document,
    IntentPlan,
    Interpretation,
    Passage,
    ScoredPassage,
    SearchEvent,
    SearchResult,
    Trajectory,
    Usage,
)
from aiotech.reasoning.scg import requirements_for
from aiotech.reasoning.tap import Reasoner, TrajectoryEvidence, select_possibilities
from aiotech.retrieval.corpus import CorpusStore
from aiotech.retrieval.reliability import ReliabilityModel
from aiotech.storage.kv import KeyValueStore
from aiotech.verification.engine import VerificationEngine

log = logging.getLogger("aiotech.pipeline")

WEB_TIMEOUT = 8.0
SINK_TIMEOUT = 3.0
MAX_GRAPH_CLAIMS = 300


class WebSearch(Protocol):
    async def search(self, query: str) -> list[Document]: ...


class GraphSink(Protocol):
    async def write(self, search_id: str, query: str, graph: dict[str, Any]) -> None: ...


@dataclass(frozen=True)
class RetrievalParams:
    """Réglages de la recherche, choisis hors de l'échantillon de test (aiotech/bench/data/SOURCES.md).

    subquery_weight  poids des entités de la question cherchées seules (fragments de la question)
    link_boost       part du score d'un passage transmise au document dont il cite le titre
    link_sources     nombre de meilleurs documents dont on suit les liens de titre
    pool_factor      profondeur du vivier de passages (× top_k) dans lequel les liens sont notés
    hop_discount     poids des passages du second saut par entités pont (palier approfondi)
    """
    subquery_weight: float = 0.5
    link_boost: float = 0.5
    link_sources: int = 2
    pool_factor: int = 3
    hop_discount: float = 0.8


@dataclass(frozen=True)
class SearchOptions:
    depth: Depth = "auto"
    use_web: bool = True
    use_llm: bool = True
    max_possibilities: int = 3


@dataclass
class _Run:
    search_id: str
    started: float
    usage: Usage = field(default_factory=Usage)
    warnings: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)

    def mark(self, name: str, since: float) -> float:
        now = time.perf_counter()
        self.timings[name] = round((now - since) * 1000, 2)
        return now


def _event(kind: str, data: dict[str, Any]) -> SearchEvent:
    return SearchEvent(type=kind, data=data)  # type: ignore[arg-type]


def _weighted(passages: Sequence[ScoredPassage], weight: float) -> list[ScoredPassage]:
    return [sp.model_copy(update={"score": round(sp.score * weight, 6)}) for sp in passages]


def _merge(results: Sequence[Sequence[ScoredPassage]], top_k: int) -> list[ScoredPassage]:
    best: dict[str, ScoredPassage] = {}
    for batch in results:
        for sp in batch:
            current = best.get(sp.passage.id)
            if current is None or sp.score > current.score:
                best[sp.passage.id] = sp
    ranked = sorted(best.values(), key=lambda sp: sp.score, reverse=True)
    return ranked[:top_k]


class SearchEngine:
    def __init__(
        self,
        *,
        corpus: CorpusStore,
        feedback: ClickFeedback,
        cache: KeyValueStore,
        guardrails: Guardrails | None = None,
        planner: RulePlanner | None = None,
        compute_gate: AdaptiveComputeGate | None = None,
        verifier: VerificationEngine | None = None,
        arg_gate: ReachabilityGate | None = None,
        web: WebSearch | None = None,
        llm: LLMTasks | None = None,
        graph_sink: GraphSink | None = None,
        reliability: ReliabilityModel | None = None,
        retrieval: RetrievalParams | None = None,
        cache_ttl: int = 600,
    ) -> None:
        self.corpus = corpus
        self.feedback = feedback
        self.cache = cache
        self.guardrails = guardrails or Guardrails()
        self.planner = planner or RulePlanner()
        self.compute_gate = compute_gate or AdaptiveComputeGate(llm_available=llm is not None)
        self.verifier = verifier or VerificationEngine()
        self.arg_gate = arg_gate or ReachabilityGate()
        self.web = web
        self.llm = llm
        self.graph_sink = graph_sink
        self.reliability = reliability or corpus.reliability
        self.params = retrieval or RetrievalParams()
        self.cache_ttl = cache_ttl

    def _cache_key(self, query: str, options: SearchOptions) -> str:
        raw = json.dumps([query, options.depth, options.use_web, options.use_llm, options.max_possibilities,
                          self.corpus.version], ensure_ascii=False)
        return "cache:" + hashlib.sha256(raw.encode()).hexdigest()

    async def search(self, query: str, options: SearchOptions | None = None) -> SearchResult:
        result: SearchResult | None = None
        async for event in self.stream(query, options):
            if event.type == "done":
                result = SearchResult.model_validate(event.data["result"])
        if result is None:
            raise RuntimeError("la recherche s'est terminée sans résultat")
        return result

    async def stream(self, query: str, options: SearchOptions | None = None) -> AsyncIterator[SearchEvent]:
        opts = options or SearchOptions()
        run = _Run(search_id=uuid.uuid4().hex[:16], started=time.perf_counter())
        try:
            async for event in self._stream(query, opts, run):
                yield event
        except Exception as exc:
            log.exception("recherche %s en échec", run.search_id)
            yield _event("error", {"search_id": run.search_id, "message": f"Erreur interne ({exc.__class__.__name__})"})
            result = SearchResult(search_id=run.search_id, query="", warnings=[*run.warnings, "Erreur interne"],
                                  usage=run.usage, metrics=self._metrics(run, False))
            yield _event("done", {"result": result.model_dump(mode="json")})

    def _metrics(self, run: _Run, cached: bool, **extra: float | int | bool | None) -> dict[str, float | int | bool | None]:
        metrics: dict[str, float | int | bool | None] = {
            "latency_ms": round((time.perf_counter() - run.started) * 1000, 2),
            "cached": cached,
        }
        metrics.update({f"{k}_ms": v for k, v in run.timings.items()})
        metrics.update(extra)
        return metrics

    async def _stream(self, query: str, opts: SearchOptions, run: _Run) -> AsyncIterator[SearchEvent]:
        screening = self.guardrails.screen(query)
        if screening.blocked:
            result = SearchResult(
                search_id=run.search_id, query="", blocked=True, block_reason=screening.reason,
                usage=run.usage, metrics=self._metrics(run, False, cost_usd=0.0, tokens=0),
            )
            yield _event("blocked", {"search_id": run.search_id, "reason": screening.reason,
                                     "rules": list(screening.rules), "tokens": 0, "cost_usd": 0.0})
            yield _event("done", {"result": result.model_dump(mode="json")})
            return
        masked = screening.text
        pii = list(screening.pii)
        yield _event("start", {"search_id": run.search_id, "query": masked,
                               "pii": [p.model_dump() for p in pii]})

        cache_key = self._cache_key(masked, opts)
        cached = await self.cache.get(cache_key)
        if cached is not None:
            result = SearchResult.model_validate_json(cached)
            result = result.model_copy(update={"search_id": run.search_id,
                                               "metrics": {**result.metrics, **self._metrics(run, True)}})
            for event in self._replay(result):
                yield event
            await self._remember_shown(result)
            yield _event("done", {"result": result.model_dump(mode="json")})
            return

        t = time.perf_counter()
        base_plan = self.planner.plan(masked, max_interpretations=4)
        budget = self.compute_gate.budget(base_plan, opts.depth, opts.use_llm)
        plan = base_plan
        if budget.use_llm_planner and self.llm is not None:
            try:
                plan, completion = await self.llm.plan(masked, base_plan, budget.effort, budget.max_interpretations)
                run.usage.add(completion.tokens_in, completion.tokens_out, completion.cost_usd)
            except LLMError as exc:
                run.warnings.append(f"Planification LLM indisponible, règles utilisées ({exc})")
        plan = plan.model_copy(update={"interpretations": plan.interpretations[: budget.max_interpretations]})
        t = run.mark("planning", t)
        yield _event("interpretations", self._interpretations_payload(plan, budget))

        web_store = CorpusStore(reliability=self.reliability, chunk_tokens=self.corpus.chunk_tokens,
                                vector_weight=self.corpus.vector_weight)
        if opts.use_web and self.web is not None:
            await self._web_retrieval(plan, web_store, run)
        t = run.mark("web", t)

        evidence_passages: dict[str, list[ScoredPassage]] = {}
        for interp in plan.interpretations:
            passages = self._retrieve(interp.subqueries, budget.top_k, web_store)
            if budget.hops > 1:
                passages = self._second_hop(plan, passages, budget.top_k, web_store)
            evidence_passages[interp.id] = passages
            yield _event("retrieval", self._retrieval_payload(interp, passages))
        context = self._retrieve([plan.query, *plan.entities], budget.top_k, web_store, self.params.subquery_weight)
        t = run.mark("retrieval", t)

        all_passages: dict[str, Passage] = {sp.passage.id: sp.passage for sp in context}
        for passages in evidence_passages.values():
            all_passages.update({sp.passage.id: sp.passage for sp in passages})
        claims_by_passage: dict[str, list[Claim]] = {
            pid: self.corpus.claims_for(pid) or web_store.claims_for(pid) for pid in all_passages
        }
        if budget.use_llm_extraction and self.llm is not None:
            await self._llm_extraction(plan, evidence_passages, claims_by_passage, budget, run)
        t = run.mark("extraction", t)

        graph = KnowledgeGraph()
        for claims in claims_by_passage.values():
            graph.add_claims(claims)
        contradictions = graph.contradictions()
        yield _event("graph", {
            "entities": len(graph.by_entity), "claims": len(graph.claims),
            "contradictions": [c.model_dump(mode="json") for c in contradictions],
        })

        reasoner = Reasoner(plan=plan, graph=graph, passages=all_passages,
                            requirements=requirements_for(graph, plan.entities), verifier=self.verifier,
                            gate=self.arg_gate)
        trajectories: list[Trajectory] = []
        for interp in plan.interpretations:
            passages = evidence_passages[interp.id]
            stored = tuple(graph.claims[c.id] for sp in passages for c in claims_by_passage.get(sp.passage.id, [])
                           if c.id in graph.claims)
            trajectory = reasoner.evaluate(TrajectoryEvidence(interp, tuple(passages), stored))
            trajectories.append(trajectory)
            yield _event("trajectory", self._trajectory_payload(trajectory))
        t = run.mark("reasoning", t)

        priors = await self.feedback.priors([i.facet for i in plan.interpretations])
        possibilities = select_possibilities(trajectories, plan.interpretations, priors,
                                             limit=max(1, min(3, opts.max_possibilities)))
        if possibilities and budget.use_llm_synthesis and self.llm is not None:
            try:
                outcome, completion = await self.llm.synthesize(masked, possibilities, all_passages, budget.effort)
                run.usage.add(completion.tokens_in, completion.tokens_out, completion.cost_usd)
                possibilities = outcome.possibilities
                run.warnings.extend(outcome.warnings)
            except LLMError as exc:
                run.warnings.append(f"Rédaction LLM indisponible, réponses gabarit ({exc})")
        if not possibilities:
            run.warnings.append("Aucune trajectoire admissible : aucune réponse ne passe la vérification")
        run.mark("synthesis", t)
        yield _event("possibilities", {"possibilities": [p.model_dump(mode="json") for p in possibilities]})

        exported = graph.export()
        exported["claims"] = exported["claims"][:MAX_GRAPH_CLAIMS]
        result = SearchResult(
            search_id=run.search_id, query=masked, pii=pii, compute=budget, plan=plan, trajectories=trajectories,
            possibilities=possibilities, contradictions=contradictions, graph=exported, warnings=run.warnings,
            usage=run.usage,
            metrics=self._metrics(run, False, passages=len(all_passages), claims=len(graph.claims),
                                  contradictions=len(contradictions)),
        )
        await self.cache.set(cache_key, result.model_dump_json(), ttl_seconds=self.cache_ttl)
        await self._remember_shown(result)
        if self.graph_sink is not None:
            try:
                await asyncio.wait_for(self.graph_sink.write(run.search_id, masked, exported), SINK_TIMEOUT)
            except Exception as exc:
                result.warnings.append(f"Graphe non enregistré dans Neo4j ({exc.__class__.__name__})")
        for warning in result.warnings:
            yield _event("warning", {"message": warning})
        yield _event("done", {"result": result.model_dump(mode="json")})

    def evidence(self, query: str, depth: Depth = "auto", top_k: int | None = None) -> tuple[list[ScoredPassage], ComputeBudget]:
        """Étape de recherche seule (requête et entités, second saut si le budget le prévoit), sans raisonnement."""
        plan = self.planner.plan(query, max_interpretations=4)
        budget = self.compute_gate.budget(plan, depth, use_llm=False)
        k = top_k or budget.top_k
        empty = CorpusStore(reliability=self.reliability, chunk_tokens=self.corpus.chunk_tokens,
                                vector_weight=self.corpus.vector_weight)
        passages = self._retrieve([plan.query, *plan.entities], k, empty, self.params.subquery_weight)
        if budget.hops > 1:
            passages = self._second_hop(plan, passages, k, empty)
        return passages, budget

    def _retrieve(self, queries: Sequence[str], top_k: int, web_store: CorpusStore,
                  secondary: float = 1.0) -> list[ScoredPassage]:
        """Recherche des requêtes puis saut par liens de titre.

        `secondary` pondère les requêtes après la première : les entités extraites de la question n'en
        sont que des fragments (`subquery_weight`), alors que les sous-requêtes d'une interprétation en
        sont des reformulations complètes (poids plein).
        """
        pool_k = top_k * self.params.pool_factor
        batches: list[list[ScoredPassage]] = []
        for n, q in enumerate(queries):
            weight = 1.0 if n == 0 else secondary
            for store in (self.corpus, web_store):
                if store is web_store and not store.passages:
                    continue
                found = store.search(q, pool_k)
                batches.append(found if weight == 1.0 else _weighted(found, weight))
        pool = _merge(batches, pool_k)
        return _merge([pool[:top_k], self._link_hop(pool, top_k, web_store)], top_k)

    def _link_hop(self, pool: list[ScoredPassage], top_k: int, web_store: CorpusStore) -> list[ScoredPassage]:
        """Saut par le graphe des documents : un des meilleurs passages cite le titre d'un autre document.

        Le document cité reçoit `link_boost` × le score du passage qui le cite, ajouté à son propre score
        pour la requête (scores de fusion rapportés au meilleur passage, voir retrieval/index.py).
        """
        own: dict[str, ScoredPassage] = {}
        for sp in pool:
            own.setdefault(sp.passage.document_id, sp)
        sources: list[ScoredPassage] = []
        for sp in pool[:top_k]:
            if all(sp.passage.document_id != s.passage.document_id for s in sources):
                sources.append(sp)
            if len(sources) == self.params.link_sources:
                break
        boosted: list[ScoredPassage] = []
        for source in sources:
            store = self.corpus if source.passage.id in self.corpus.passages else web_store
            for document_id in store.linked_documents(source.passage.text):
                if document_id == source.passage.document_id:
                    continue
                current = own.get(document_id)
                target = current.passage if current is not None else store.first_passage(document_id)
                if target is None:
                    continue
                base = current.score if current is not None else 0.0
                boosted.append(ScoredPassage(passage=target, score=round(base + self.params.link_boost * source.score, 6),
                                             bm25_rank=current.bm25_rank if current else None,
                                             vector_rank=current.vector_rank if current else None))
        return boosted

    def _second_hop(self, plan: IntentPlan, first: list[ScoredPassage], top_k: int,
                    web_store: CorpusStore) -> list[ScoredPassage]:
        """Second saut : entités pont trouvées dans les meilleurs passages, absentes de la requête."""
        query_stems = set(content_words(plan.query))
        bridges: list[str] = []
        for sp in first[:3]:
            for entity in query_entities(sp.passage.text):
                stems = set(content_words(entity))
                if stems and not stems <= query_stems and entity not in bridges:
                    bridges.append(entity)
        if not bridges:
            return first
        terms = " ".join(plan.key_terms[:4])
        hop = self._retrieve([f"{b} {terms}".strip() for b in bridges[:2]], top_k, web_store)
        return _merge([first, _weighted(hop, self.params.hop_discount)], top_k)

    async def _web_retrieval(self, plan: IntentPlan, store: CorpusStore, run: _Run) -> None:
        assert self.web is not None
        queries = list(dict.fromkeys([plan.query, *(i.subqueries[0] for i in plan.interpretations if i.subqueries)]))
        tasks = [asyncio.wait_for(self.web.search(q), WEB_TIMEOUT) for q in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        failures = 0
        rejected = 0
        for outcome in results:
            if isinstance(outcome, BaseException):
                failures += 1
                continue
            for document in outcome:
                if document.id in store.documents:
                    continue
                if not self.guardrails.passage_is_safe(document.title + " " + document.text):
                    rejected += 1
                    continue
                safe = document.model_copy(update={"text": self.guardrails.mask(document.text)})
                store.add(safe, persist=False)
        if failures:
            run.warnings.append(f"Recherche web indisponible pour {failures} requête(s) sur {len(queries)}")
        if rejected:
            run.warnings.append(f"{rejected} résultat(s) web écarté(s) : tentative d'injection indirecte")

    async def _llm_extraction(
        self,
        plan: IntentPlan,
        evidence: dict[str, list[ScoredPassage]],
        claims_by_passage: dict[str, list[Claim]],
        budget: ComputeBudget,
        run: _Run,
    ) -> None:
        assert self.llm is not None
        ordered: dict[str, Passage] = {}
        depth = 0
        while len(ordered) < sum(len(v) for v in evidence.values()) and depth < budget.top_k:
            for passages in evidence.values():
                if depth < len(passages):
                    ordered.setdefault(passages[depth].passage.id, passages[depth].passage)
            depth += 1
        safe = [p for p in ordered.values() if self.guardrails.passage_is_safe(p.text)]
        try:
            claims, completion = await self.llm.extract(plan.query, safe, budget.context_tokens, budget.effort)
        except LLMError as exc:
            run.warnings.append(f"Extraction LLM indisponible, règles seules ({exc})")
            return
        if completion is not None:
            run.usage.add(completion.tokens_in, completion.tokens_out, completion.cost_usd)
        for claim in claims:
            claims_by_passage.setdefault(claim.passage_id, []).append(claim)

    async def _remember_shown(self, result: SearchResult) -> None:
        if result.plan is None:
            return
        facets = {i.id: i.facet for i in result.plan.interpretations}
        shown = {p.interpretation_id: facets.get(p.interpretation_id, p.interpretation_id) for p in result.possibilities}
        await self.cache.set(f"shown:{result.search_id}", json.dumps(shown), ttl_seconds=86400)

    async def record_click(self, search_id: str, interpretation_id: str) -> dict[str, float]:
        raw = await self.cache.get(f"shown:{search_id}")
        if raw is None:
            raise KeyError("recherche inconnue ou expirée")
        shown: dict[str, str] = json.loads(raw)
        if interpretation_id not in shown:
            raise KeyError("cette interprétation n'a pas été proposée pour cette recherche")
        return await self.feedback.record(shown[interpretation_id], list(shown.values()))

    @staticmethod
    def _interpretations_payload(plan: IntentPlan, budget: ComputeBudget) -> dict[str, Any]:
        return {
            "question_type": plan.question_type,
            "objective": plan.objective,
            "planner": plan.planner,
            "constraints": [c.label for c in plan.constraints],
            "entities": list(plan.entities),
            "compute": budget.model_dump(mode="json"),
            "neurons": [_neuron(i) for i in plan.interpretations],
        }

    @staticmethod
    def _retrieval_payload(interp: Interpretation, passages: Sequence[ScoredPassage]) -> dict[str, Any]:
        return {
            "interpretation_id": interp.id,
            "passages": [
                {"id": sp.passage.id, "title": sp.passage.source.title, "url": sp.passage.source.url,
                 "origin": sp.passage.source.origin, "reliability": sp.passage.source.reliability,
                 "score": sp.score}
                for sp in passages
            ],
        }

    @staticmethod
    def _trajectory_payload(trajectory: Trajectory) -> dict[str, Any]:
        return {
            "interpretation_id": trajectory.interpretation_id,
            "admissible": trajectory.admissible,
            "rejection_reason": trajectory.rejection_reason,
            "best": None if trajectory.best is None else {
                "label": trajectory.best.label, "reliability": trajectory.best.score.reliability,
            },
            "candidates": [
                {"label": c.label, "rejected": c.rejected, "reason": c.rejection_reason,
                 "admissibility": c.score.admissibility, "reliability": c.score.reliability}
                for c in trajectory.candidates[:8]
            ],
        }

    def _replay(self, result: SearchResult) -> list[SearchEvent]:
        events: list[SearchEvent] = []
        if result.plan is not None and result.compute is not None:
            events.append(_event("interpretations", self._interpretations_payload(result.plan, result.compute)))
        for trajectory in result.trajectories:
            events.append(_event("retrieval", {"interpretation_id": trajectory.interpretation_id, "passages": [],
                                               "count": trajectory.passages}))
        events.append(_event("graph", {"entities": len(result.graph.get("entities", [])),
                                       "claims": len(result.graph.get("claims", [])),
                                       "contradictions": [c.model_dump(mode="json") for c in result.contradictions]}))
        events.extend(_event("trajectory", self._trajectory_payload(t)) for t in result.trajectories)
        events.append(_event("possibilities",
                             {"possibilities": [p.model_dump(mode="json") for p in result.possibilities]}))
        return events


def _neuron(interp: Interpretation) -> dict[str, Any]:
    return {"id": interp.id, "label": interp.label, "facet": interp.facet, "rationale": interp.rationale,
            "plausibility": interp.plausibility, "subqueries": list(interp.subqueries)}
