"""
Configuration par variables d'environnement (voir .env.example à la racine du dépôt).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ProviderName = Literal["anthropic", "litellm", "none"]


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "oui", "on"}


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


def _str(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _path(name: str) -> Path | None:
    raw = _str(name)
    return Path(raw) if raw else None


@dataclass(frozen=True)
class Settings:
    llm_provider: ProviderName
    llm_model: str
    llm_fast_model: str
    llm_fallback_models: tuple[str, ...]
    llm_api_base: str
    anthropic_api_key: str
    anthropic_base_url: str
    anthropic_server_fallbacks: bool
    api_keys: str
    admin_token: str
    require_api_key: bool
    quota_per_minute: int
    quota_per_day: int
    redis_url: str
    neo4j_url: str
    neo4j_user: str
    neo4j_password: str
    searxng_url: str
    corpus_dir: Path | None
    docs_dir: Path | None
    load_demo: bool
    source_reliability: str
    max_context_tokens: int
    cache_ttl: int
    cors_origins: tuple[str, ...]

    @classmethod
    def from_env(cls) -> Settings:
        providers: dict[str, ProviderName] = {"anthropic": "anthropic", "litellm": "litellm", "none": "none"}
        provider = providers.get(_str("AIOTECH_LLM_PROVIDER", "anthropic").lower())
        if provider is None:
            raise ValueError("AIOTECH_LLM_PROVIDER doit valoir anthropic, litellm ou none")
        default_model = "claude-opus-5-5" if provider == "anthropic" else ""
        api_keys = _str("AIOTECH_API_KEYS")
        return cls(
            llm_provider=provider,
            llm_model=_str("AIOTECH_LLM_MODEL", default_model) or default_model,
            llm_fast_model=_str("AIOTECH_LLM_FAST_MODEL"),
            llm_fallback_models=tuple(m.strip() for m in _str("AIOTECH_LLM_FALLBACK_MODELS").split(",") if m.strip()),
            llm_api_base=_str("AIOTECH_LLM_API_BASE"),
            anthropic_api_key=_str("ANTHROPIC_API_KEY"),
            anthropic_base_url=_str("AIOTECH_ANTHROPIC_BASE_URL"),
            anthropic_server_fallbacks=_bool("AIOTECH_ANTHROPIC_SERVER_FALLBACKS", True),
            api_keys=api_keys,
            admin_token=_str("AIOTECH_ADMIN_TOKEN"),
            require_api_key=_bool("AIOTECH_REQUIRE_API_KEY", bool(api_keys)),
            quota_per_minute=_int("AIOTECH_QUOTA_PER_MINUTE", 30),
            quota_per_day=_int("AIOTECH_QUOTA_PER_DAY", 2000),
            redis_url=_str("AIOTECH_REDIS_URL"),
            neo4j_url=_str("AIOTECH_NEO4J_URL"),
            neo4j_user=_str("AIOTECH_NEO4J_USER", "neo4j"),
            neo4j_password=_str("AIOTECH_NEO4J_PASSWORD"),
            searxng_url=_str("AIOTECH_SEARXNG_URL"),
            corpus_dir=_path("AIOTECH_CORPUS_DIR"),
            docs_dir=_path("AIOTECH_DOCS_DIR"),
            load_demo=_bool("AIOTECH_LOAD_DEMO", True),
            source_reliability=_str("AIOTECH_SOURCE_RELIABILITY"),
            max_context_tokens=_int("AIOTECH_MAX_CONTEXT_TOKENS", 4000),
            cache_ttl=_int("AIOTECH_CACHE_TTL", 600),
            cors_origins=tuple(o.strip() for o in _str("AIOTECH_CORS_ORIGINS").split(",") if o.strip()),
        )
