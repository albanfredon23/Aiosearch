"""
API HTTP d'AIOTECH Search.

    GET  /health                       état des composants (sans secret)
    POST /v1/search                    résultat complet
    GET  /v1/search/stream?q=...       étapes en Server-Sent Events (interface 3D, EventSource)
    POST /v1/search/stream             idem pour les clients HTTP
    POST /v1/feedback                  clic sur une réponse (départage à fiabilité égale)
    GET  /v1/models                    modèles exposés, format OpenAI
    POST /v1/chat/completions          compatible OpenAI (stream ou non) : tout client LLM s'y branche
    GET  /v1/tools?format=...          schémas d'outils (anthropic | openai) pour l'appel d'outils
    POST /v1/tools/{name}              exécution d'un outil confiné
    /mcp                               serveur MCP (HTTP streamable), mêmes outils
    /admin/...                         clé d'API valide ET jeton d'administration
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from aiotech import __version__
from aiotech.core.adaptive import Depth, parse_depth
from aiotech.gateway.auth import ApiPrincipal, AuthError, digest
from aiotech.gateway.tools import ToolError
from aiotech.llm.base import Completion, LLMError, StreamDelta
from aiotech.models import Document, SearchResult
from aiotech.pipeline import SearchOptions
from aiotech.runtime import Runtime, build_runtime, summarize, template_answer
from aiotech.settings import Settings

log = logging.getLogger("aiotech.api")
MODEL_ID = "aiotech-search"


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=4000)
    depth: Depth = "auto"
    use_web: bool = True
    use_llm: bool = True
    max_possibilities: int = Field(default=3, ge=1, le=3)

    def options(self) -> SearchOptions:
        return SearchOptions(depth=self.depth, use_web=self.use_web, use_llm=self.use_llm,
                             max_possibilities=self.max_possibilities)


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    search_id: str = Field(min_length=4, max_length=64)
    interpretation_id: str = Field(min_length=1, max_length=16)


class ChatMessageIn(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Literal["system", "user", "assistant", "tool", "developer"]
    content: str | list[dict[str, Any]] | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = MODEL_ID
    messages: list[ChatMessageIn] = Field(min_length=1, max_length=200)
    stream: bool = False
    max_tokens: int | None = Field(default=None, ge=1, le=8000)


class DocumentIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,80}$")
    title: str = Field(min_length=1, max_length=300)
    text: str = Field(min_length=1, max_length=200_000)
    url: str | None = Field(default=None, max_length=2000)
    reliability: float | None = Field(default=None, ge=0.0, le=1.0)
    metadata: dict[str, str] = Field(default_factory=dict)


def _text_of(message: ChatMessageIn) -> str:
    if isinstance(message.content, str):
        return message.content
    if isinstance(message.content, list):
        return " ".join(str(part.get("text", "")) for part in message.content if part.get("type") == "text")
    return ""


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _presented_key(request: Request) -> str | None:
    key = request.headers.get("x-api-key")
    if key:
        return key.strip()
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return None


def create_app(settings: Settings | None = None, runtime: Runtime | None = None) -> FastAPI:
    holder: dict[str, Runtime] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        rt = runtime or await build_runtime(settings)
        holder["rt"] = rt
        mcp_app = None
        try:
            from aiotech.mcp_server import build_mcp

            mcp = build_mcp(rt)
            mcp_app = mcp.streamable_http_app(streamable_http_path="/", stateless_http=True, json_response=True,
                                              host="0.0.0.0")
            app.mount("/mcp", _protected(mcp_app, rt))
            async with mcp.session_manager.run():
                yield
        except ImportError:
            log.warning("paquet mcp absent : /mcp désactivé")
            yield
        finally:
            await rt.close()

    app = FastAPI(title="AIOTECH Search", version=__version__, lifespan=lifespan,
                  docs_url="/docs", redoc_url=None, openapi_url="/openapi.json")
    cfg_origins = (settings.cors_origins if settings else ()) or (runtime.settings.cors_origins if runtime else ())
    if cfg_origins:
        app.add_middleware(CORSMiddleware, allow_origins=list(cfg_origins), allow_methods=["GET", "POST"],
                           allow_headers=["content-type", "x-api-key", "authorization"])

    @app.middleware("http")
    async def security_headers(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("Cache-Control", "no-store")
        return response

    @app.exception_handler(AuthError)
    async def auth_error(_: Request, exc: AuthError) -> JSONResponse:
        return JSONResponse({"error": {"type": "authentication_error", "message": exc.message}}, status_code=exc.status)

    def rt() -> Runtime:
        return holder["rt"]

    async def principal(request: Request) -> ApiPrincipal:
        """Clé exigée si AIOTECH_REQUIRE_API_KEY ; sinon clé facultative, quotas par adresse IP."""
        r = rt()
        presented = _presented_key(request)
        client = request.client.host if request.client else "inconnu"
        if r.settings.require_api_key or (presented is not None and r.keys.enabled):
            who = r.keys.authenticate(presented)
        else:
            who = ApiPrincipal(name="anonyme", key_id=digest(client)[:12])
        bucket = f"{who.key_id}:{digest(client)[:12]}" if who.name in r.settings.per_client_keys else who.key_id
        decision = await r.quotas.consume(bucket)
        if not decision.allowed:
            raise HTTPException(status_code=429, detail="Quota dépassé", headers={"Retry-After": str(decision.retry_after)})
        return who

    async def admin(request: Request, x_admin_token: str | None = Header(default=None)) -> ApiPrincipal:
        r = rt()
        if not r.keys.enabled:
            raise AuthError(403, "Administration désactivée : aucune clé d'API configurée")
        who = r.keys.authenticate(_presented_key(request))
        if who.name in r.settings.per_client_keys:
            raise AuthError(403, "Cette clé est réservée à l'interface web")
        r.keys.authorize_admin(x_admin_token)
        return who

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return await rt().health()

    @app.post("/v1/search")
    async def search(body: SearchRequest, _: ApiPrincipal = Depends(principal)) -> dict[str, Any]:
        result = await rt().engine.search(body.query, body.options())
        return result.model_dump(mode="json")

    async def event_stream(query: str, options: SearchOptions) -> AsyncIterator[str]:
        async for event in rt().engine.stream(query, options):
            yield _sse(event.type, event.data)

    @app.get("/v1/search/stream")
    async def search_stream_get(
        q: str = Query(min_length=1, max_length=4000),
        depth: str = Query(default="auto"),
        web: bool = Query(default=True),
        llm: bool = Query(default=True),
        _: ApiPrincipal = Depends(principal),
    ) -> StreamingResponse:
        options = SearchOptions(depth=parse_depth(depth), use_web=web, use_llm=llm)
        return StreamingResponse(event_stream(q, options), media_type="text/event-stream",
                                 headers={"X-Accel-Buffering": "no"})

    @app.post("/v1/search/stream")
    async def search_stream_post(body: SearchRequest, _: ApiPrincipal = Depends(principal)) -> StreamingResponse:
        return StreamingResponse(event_stream(body.query, body.options()), media_type="text/event-stream",
                                 headers={"X-Accel-Buffering": "no"})

    @app.post("/v1/feedback")
    async def feedback(body: FeedbackRequest, _: ApiPrincipal = Depends(principal)) -> dict[str, Any]:
        try:
            priors = await rt().engine.record_click(body.search_id, body.interpretation_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc.args[0])) from exc
        return {"recorded": True, "click_priors": priors}

    @app.get("/v1/models")
    async def models(_: ApiPrincipal = Depends(principal)) -> dict[str, Any]:
        return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "created": 0, "owned_by": "aiotech"}]}

    @app.post("/v1/chat/completions", response_model=None)
    async def chat(body: ChatCompletionRequest, _: ApiPrincipal = Depends(principal)) -> dict[str, Any] | StreamingResponse:
        query = next((_text_of(m) for m in reversed(body.messages) if m.role == "user"), "").strip()
        if not query:
            raise HTTPException(status_code=400, detail="Aucun message utilisateur")
        r = rt()
        result = await r.engine.search(query, SearchOptions())
        completion_id = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())
        if body.stream:
            return StreamingResponse(_chat_stream(r, result, completion_id, created, body.max_tokens),
                                     media_type="text/event-stream", headers={"X-Accel-Buffering": "no"})
        text, completion = await _chat_answer(r, result, body.max_tokens)
        usage_in = result.usage.tokens_in + (completion.tokens_in if completion else 0)
        usage_out = result.usage.tokens_out + (completion.tokens_out if completion else 0)
        return {
            "id": completion_id, "object": "chat.completion", "created": created, "model": MODEL_ID,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": usage_in, "completion_tokens": usage_out, "total_tokens": usage_in + usage_out},
            "aiotech": summarize(result),
        }

    @app.get("/v1/tools")
    async def tools(format: str = Query(default="anthropic"), _: ApiPrincipal = Depends(principal)) -> dict[str, Any]:
        registry = rt().tools
        if format == "openai":
            return {"tools": registry.openai_schemas()}
        return {"tools": registry.anthropic_schemas()}

    @app.post("/v1/tools/{name}")
    async def call_tool(name: str, arguments: dict[str, Any], _: ApiPrincipal = Depends(principal)) -> dict[str, Any]:
        try:
            return {"tool": name, "result": await rt().tools.call(name, arguments)}
        except ToolError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/admin/stats")
    async def admin_stats(_: ApiPrincipal = Depends(admin)) -> dict[str, Any]:
        r = rt()
        return {"corpus": r.corpus.stats(), "llm": r.router.status() if r.router else [], "notes": r.notes,
                "tools": list(r.tools.tools)}

    @app.post("/admin/documents")
    async def admin_add(body: DocumentIn, _: ApiPrincipal = Depends(admin)) -> dict[str, Any]:
        document = Document(**body.model_dump(), origin="corpus")
        try:
            passages = rt().corpus.add(document)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"id": document.id, "passages": len(passages), "corpus": rt().corpus.stats()}

    @app.delete("/admin/documents/{document_id}")
    async def admin_remove(document_id: str, _: ApiPrincipal = Depends(admin)) -> dict[str, Any]:
        if not rt().corpus.remove(document_id):
            raise HTTPException(status_code=404, detail="Document inconnu")
        return {"removed": document_id, "corpus": rt().corpus.stats()}

    return app


def _protected(inner: Any, runtime: Runtime) -> Any:
    """Enveloppe ASGI : le serveur MCP HTTP exige la même clé d'API que le reste de l'API."""

    async def app(scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] == "http" and runtime.settings.require_api_key:
            headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
            presented = headers.get("x-api-key") or (
                headers.get("authorization", "")[7:].strip()
                if headers.get("authorization", "").lower().startswith("bearer ") else None
            )
            try:
                runtime.keys.authenticate(presented)
            except AuthError as exc:
                body = json.dumps({"error": {"type": "authentication_error", "message": exc.message}}).encode()
                await send({"type": "http.response.start", "status": exc.status,
                            "headers": [(b"content-type", b"application/json")]})
                await send({"type": "http.response.body", "body": body})
                return
        await inner(scope, receive, send)

    return app


async def _chat_answer(r: Runtime, result: SearchResult, max_tokens: int | None) -> tuple[str, Completion | None]:
    if result.blocked or r.llm is None or not result.possibilities:
        return template_answer(result), None
    effort = result.compute.effort if result.compute else "medium"
    parts: list[str] = []
    final: Completion | None = None
    try:
        async for item in r.llm.stream_chat(result.query, result.possibilities, effort, max_tokens or 1200):
            if isinstance(item, StreamDelta):
                parts.append(item.text)
            else:
                final = item
    except LLMError:
        return template_answer(result), None
    return ("".join(parts).strip() or template_answer(result)), final


async def _chat_stream(r: Runtime, result: SearchResult, completion_id: str, created: int,
                       max_tokens: int | None) -> AsyncIterator[str]:
    def chunk(delta: dict[str, Any], finish: str | None = None, extra: dict[str, Any] | None = None) -> str:
        payload: dict[str, Any] = {"id": completion_id, "object": "chat.completion.chunk", "created": created,
                                   "model": MODEL_ID,
                                   "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
        if extra:
            payload.update(extra)
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    yield chunk({"role": "assistant", "content": ""})
    if result.blocked or r.llm is None or not result.possibilities:
        yield chunk({"content": template_answer(result)})
    else:
        effort = result.compute.effort if result.compute else "medium"
        emitted = False
        try:
            async for item in r.llm.stream_chat(result.query, result.possibilities, effort, max_tokens or 1200):
                if isinstance(item, StreamDelta) and item.text:
                    emitted = True
                    yield chunk({"content": item.text})
        except LLMError:
            note = "\n\n[Rédaction interrompue ; réponses vérifiées ci-dessous]\n\n" if emitted else ""
            yield chunk({"content": note + template_answer(result)})
    yield chunk({}, finish="stop", extra={"aiotech": summarize(result)})
    yield "data: [DONE]\n\n"

