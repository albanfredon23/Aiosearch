"""
Stockage clé-valeur des quotas, du cache de réponses et des retours de clics.

Deux implémentations au même contrat asynchrone :
    MemoryKV  processus unique (tests, mode hors ligne) ;
    RedisKV   partagé entre plusieurs instances du moteur (Docker : service redis interne).
"""
from __future__ import annotations

import asyncio
import time
from typing import Protocol

from redis.asyncio import Redis


class KeyValueStore(Protocol):
    async def get(self, key: str) -> str | None: ...

    async def set(self, key: str, value: str, ttl_seconds: int | None = None) -> None: ...

    async def incr(self, key: str, amount: int = 1, ttl_seconds: int | None = None) -> int: ...

    async def delete(self, key: str) -> None: ...

    async def ping(self) -> bool: ...

    async def close(self) -> None: ...


class MemoryKV:
    """Dictionnaire en mémoire avec expiration paresseuse."""

    def __init__(self) -> None:
        self._data: dict[str, tuple[str, float | None]] = {}
        self._lock = asyncio.Lock()

    def _alive(self, key: str) -> str | None:
        item = self._data.get(key)
        if item is None:
            return None
        value, expires = item
        if expires is not None and expires <= time.monotonic():
            del self._data[key]
            return None
        return value

    async def get(self, key: str) -> str | None:
        async with self._lock:
            return self._alive(key)

    async def set(self, key: str, value: str, ttl_seconds: int | None = None) -> None:
        async with self._lock:
            expires = time.monotonic() + ttl_seconds if ttl_seconds else None
            self._data[key] = (value, expires)

    async def incr(self, key: str, amount: int = 1, ttl_seconds: int | None = None) -> int:
        async with self._lock:
            current = self._alive(key)
            if current is None:
                expires = time.monotonic() + ttl_seconds if ttl_seconds else None
                value = amount
            else:
                expires = self._data[key][1]
                value = int(current) + amount
            self._data[key] = (str(value), expires)
            return value

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._data.pop(key, None)

    async def ping(self) -> bool:
        return True

    async def close(self) -> None:
        return None


class RedisKV:
    """Redis asynchrone ; l'expiration n'est posée qu'à la création d'un compteur (fenêtre fixe)."""

    def __init__(self, url: str, prefix: str = "aiotech:") -> None:
        self._redis: Redis = Redis.from_url(url, decode_responses=True, socket_timeout=2.0)
        self._prefix = prefix

    async def get(self, key: str) -> str | None:
        value = await self._redis.get(self._prefix + key)
        return None if value is None else str(value)

    async def set(self, key: str, value: str, ttl_seconds: int | None = None) -> None:
        await self._redis.set(self._prefix + key, value, ex=ttl_seconds)

    async def incr(self, key: str, amount: int = 1, ttl_seconds: int | None = None) -> int:
        full = self._prefix + key
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.incrby(full, amount)
            if ttl_seconds:
                pipe.expire(full, ttl_seconds, nx=True)
            results = await pipe.execute()
        return int(results[0])

    async def delete(self, key: str) -> None:
        await self._redis.delete(self._prefix + key)

    async def ping(self) -> bool:
        try:
            return bool(await self._redis.ping())
        except Exception:
            return False

    async def close(self) -> None:
        await self._redis.aclose()
