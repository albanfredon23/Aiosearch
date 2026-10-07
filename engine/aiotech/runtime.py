"""
Assemblage des composants à partir de la configuration (API HTTP, serveur MCP, benchmark).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from importlib import resources
from typing import Any

import httpx

from aiotech import __version__
from aiotech.core.adaptive import AdaptiveComputeGate, parse_depth
from aiotech.core.feedback import ClickFeedback
from aiotech.gateway.auth import ApiKeyStore, QuotaManager
from aiotech.gateway.guardrails import Guardrails
from aiotech.gateway.tools import FileJail, ToolRegistry, ToolSpec, default_registry
from aiotech.graph.neo4j_sink import Neo4jSink
from aiotech.llm.anthropic_provider import AnthropicProvider
from aiotech.llm.base import LLMError, LLMProvider
from aiotech.llm.litellm_provider import LiteLLMProvider
from aiotech.llm.router import LLMRouter
from aiotech.llm.tasks import LLMTasks
from aiotech.models import Document, SearchResult
from aiotech.pipeline import SearchEngine, SearchOptions
from aiotech.retrieval.corpus import CorpusStore
from aiotech.retrieval.reliability import ReliabilityModel
from aiotech.retrieval.web import SearxngClient
from aiotech.settings import Settings
from aiotech.storage.kv import KeyValueStore, MemoryKV, RedisKV

log = logging.getLogger("aiotech.runtime")


def demo_documents() -> list[Document]:
    raw = resources.files("aiotech.data").joinpath("demo_corpus.json").read_text("utf-8")
    return [Document.model_validate(item) for item in json.loads(raw)]


def _provider(kind: str, model: str, settings: Settings) -> LLMProvider:
    if kind == "anthropic":
        return AnthropicProvider(api_key=settings.anthropic_api_key, model=model,
                                 base_url=settings.anthropic_base_url or None,
                                 server_fallbacks=settings.anthropic_server_fallbacks)
    return LiteLLMProvider(model=model, api_base=settings.llm_api_base or None)


def build_routers(settings: Settings) -> tuple[LLMRouter | None, LLMRouter | None, list[str]]:
    """Routeur principal, routeur rapide (extraction) et avertissements de configuration."""
    notes: list[str] = []
    if settings.llm_provider == "none":
        return None, None, ["LLM désactivé : mode hors ligne (règles et réponses gabarit)"]
    if settings.llm_provider == "anthropic" and not settings.anthropic_api_key:
        return None, None, ["ANTHROPIC_API_KEY absente : mode hors ligne (règles et réponses gabarit)"]
    if not settings.llm_model:
        return None, None, ["AIOTECH_LLM_MODEL absent : mode hors ligne"]
    providers: list[LLMProvider] = []
    try:
        providers.append(_provider(settings.llm_provider, settings.llm_model, settings))
    except LLMError as exc:
        return None, None, [f"LLM indisponible : {exc}"]
    for model in settings.llm_fallback_models:
        try:
            providers.append(LiteLLMProvider(model=model, api_base=settings.llm_api_base or None))
        except LLMError as exc:
            notes.append(f"Repli {model} ignoré : {exc}")
    fast: LLMRouter | None = None
    if settings.llm_fast_model:
        try:
            fast = LLMRouter([_provider(settings.llm_provider, settings.llm_fast_model, settings)])
        except LLMError as exc:
            notes.append(f"Modèle rapide ignoré : {exc}")
    return LLMRouter(providers), fast, notes


@dataclass
class Runtime:
    settings: Settings
    kv: KeyValueStore
    corpus: CorpusStore
    engine: SearchEngine
    keys: ApiKeyStore
    quotas: QuotaManager
    tools: ToolRegistry
    router: LLMRouter | None
    llm: LLMTasks | None
    sink: Neo4jSink | None
    http: httpx.AsyncClient | None
    notes: list[str] = field(default_factory=list)

    async def close(self) -> None:
        if self.http is not None:
            await self.http.aclose()
        if self.sink is not None:
            await self.sink.close()
        await self.kv.close()

    async def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "llm": self.router.primary_model if self.router else None,
            "web": self.engine.web is not None,
            "redis": await self.kv.ping() if isinstance(self.kv, RedisKV) else None,
            "neo4j": await self.sink.ping() if self.sink is not None else None,
            "corpus": self.corpus.stats(),
        }


def summarize(result: SearchResult) -> dict[str, Any]:
    """Résultat compact pour les outils, le MCP et l'API compatible OpenAI."""
    return {
        "search_id": result.search_id,
        "query": result.query,
        "blocked": result.blocked,
        "block_reason": result.block_reason,
        "answers": [
            {
                "rank": p.rank,
                "interpretation": p.interpretation_label,
                "title": p.title,
                "answer": p.answer,
                "reliability_pct": p.reliability_pct,
                "claims": [{"status": c.status.value, "text": c.text, "reason": c.reason} for c in p.claims],
                "sources": [{"title": s.title, "url": s.url, "reliability": s.reliability} for s in p.sources],
            }
            for p in result.possibilities
        ],
        "contradictions": [
            {"entity": c.entity_label, "attribute": c.attribute, "values": [g.value_text for g in c.groups],
             "resolved": c.resolved, "winner": c.winner}
            for c in result.contradictions
        ],
        "warnings": result.warnings,
        "usage": result.usage.model_dump(),
    }


def template_answer(result: SearchResult) -> str:
    if result.blocked:
        return f"Requête refusée : {result.block_reason}."
    if not result.possibilities:
        return "Aucune réponse ne passe la vérification : les sources disponibles ne permettent pas de répondre de façon fiable."
    lines = [f"Réponses vérifiées pour « {result.query} » :", ""]
    for p in result.possibilities:
        lines.append(f"{p.rank}. {p.title} (fiabilité {p.reliability_pct} %, lecture « {p.interpretation_label} »)")
        lines.append(f"   {p.answer}")
        for c in p.claims[:6]:
            lines.append(f"   [{c.status.value}] {c.text}")
        sources = ", ".join(s.title for s in p.sources[:3])
        if sources:
            lines.append(f"   Sources : {sources}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _search_tool(engine: SearchEngine) -> ToolSpec:
    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query", ""))
        options = SearchOptions(depth=parse_depth(str(args.get("depth", "auto"))),
                                use_web=bool(args.get("use_web", True)))
        return summarize(await engine.search(query, options))

    return ToolSpec(
        name="aiotech_search",
        description=(
            "Recherche vérifiée AIOTECH : 2 ou 3 réponses classées par fiabilité, chaque affirmation marquée "
            "[FAIT], [INFÉRENCE], [INCERTAIN] ou [NON VÉRIFIÉ], avec ses sources et les contradictions détectées."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "maxLength": 2000},
                "depth": {"type": "string", "enum": ["auto", "light", "standard", "deep"]},
                "use_web": {"type": "boolean"},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        handler=handler,
    )


async def build_runtime(settings: Settings | None = None) -> Runtime:
    cfg = settings or Settings.from_env()
    kv: KeyValueStore = RedisKV(cfg.redis_url) if cfg.redis_url else MemoryKV()
    reliability = ReliabilityModel.from_json(cfg.source_reliability or None)
    corpus = CorpusStore(reliability=reliability, persist_dir=cfg.corpus_dir)
    if cfg.corpus_dir is not None:
        corpus.load_dir(cfg.corpus_dir)
    if cfg.load_demo:
        for document in demo_documents():
            if document.id not in corpus.documents:
                corpus.add(document, persist=False)
    router, fast, notes = build_routers(cfg)
    llm = LLMTasks(router, fast) if router is not None else None
    http = httpx.AsyncClient(timeout=httpx.Timeout(8.0, connect=3.0), follow_redirects=False) if cfg.searxng_url else None
    web = SearxngClient(base_url=cfg.searxng_url, http=http) if http is not None else None
    sink = Neo4jSink(cfg.neo4j_url, cfg.neo4j_user, cfg.neo4j_password) if cfg.neo4j_url and cfg.neo4j_password else None
    engine = SearchEngine(
        corpus=corpus,
        feedback=ClickFeedback(kv),
        cache=kv,
        guardrails=Guardrails(),
        compute_gate=AdaptiveComputeGate(max_context_tokens=cfg.max_context_tokens, llm_available=llm is not None),
        web=web,
        llm=llm,
        graph_sink=sink,
        reliability=reliability,
        cache_ttl=cfg.cache_ttl,
    )
    jail = FileJail(cfg.docs_dir) if cfg.docs_dir is not None and cfg.docs_dir.is_dir() else None
    tools = default_registry(jail)
    tools.register(_search_tool(engine))
    extra: dict[str, str] = {}
    if cfg.web_api_key_file is not None:
        try:
            web_key = cfg.web_api_key_file.read_text("utf-8").strip()
        except OSError as exc:
            notes.append(f"clé de l'interface web illisible ({exc.strerror}) : interface désactivée")
        else:
            if web_key:
                extra["web"] = web_key
    keys = ApiKeyStore.from_env(cfg.api_keys, cfg.admin_token or None, extra)
    quotas = QuotaManager(kv, cfg.quota_per_minute, cfg.quota_per_day)
    for note in notes:
        log.warning(note)
    return Runtime(settings=cfg, kv=kv, corpus=corpus, engine=engine, keys=keys, quotas=quotas, tools=tools,
                   router=router, llm=llm, sink=sink, http=http, notes=notes)
