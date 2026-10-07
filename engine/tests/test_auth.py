from __future__ import annotations

from pathlib import Path

import pytest

from aiotech.gateway.auth import ApiKeyStore, AuthError, QuotaManager
from aiotech.storage.kv import MemoryKV


def test_api_keys_are_checked() -> None:
    store = ApiKeyStore.from_env("web:cle-web-0123456789abcdef,partenaire:cle-part-0123456789abcdef", "admin-0123456789")
    assert store.authenticate("cle-web-0123456789abcdef").name == "web"
    with pytest.raises(AuthError):
        store.authenticate("mauvaise-cle-0123456789")
    with pytest.raises(AuthError):
        store.authenticate(None)


def test_short_keys_are_rejected() -> None:
    with pytest.raises(ValueError):
        ApiKeyStore.from_env("web:court", None)


def test_admin_requires_token_and_is_closed_by_default() -> None:
    store = ApiKeyStore.from_env("web:cle-web-0123456789abcdef", "admin-0123456789")
    store.authorize_admin("admin-0123456789")
    with pytest.raises(AuthError):
        store.authorize_admin("faux")
    closed = ApiKeyStore.from_env("web:cle-web-0123456789abcdef", None)
    with pytest.raises(AuthError) as exc:
        closed.authorize_admin("nimporte")
    assert exc.value.status == 403


async def test_quota_per_minute_and_day() -> None:
    quotas = QuotaManager(MemoryKV(), per_minute=2, per_day=3)
    now = 1_000_000.0
    assert (await quotas.consume("k", now)).allowed
    assert (await quotas.consume("k", now)).allowed
    refused = await quotas.consume("k", now)
    assert not refused.allowed and refused.retry_after > 0
    assert not (await quotas.consume("k", now + 60)).allowed
    assert (await quotas.consume("autre", now)).allowed


def test_keygen_creates_once_and_never_overwrites(tmp_path: Path) -> None:
    from aiotech.keygen import ensure

    assert sorted(ensure(tmp_path)) == ["searxng_secret", "web_key"]
    first = (tmp_path / "web_key").read_text()
    assert len(first) >= 40
    assert ensure(tmp_path) == []
    assert (tmp_path / "web_key").read_text() == first
