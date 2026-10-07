"""
Authentification et quotas.

    clés d'API     AIOTECH_API_KEYS="nom:clé,nom2:clé2" ; seules leurs empreintes SHA-256 sont
                   gardées en mémoire, comparées en temps constant (hmac.compare_digest) ;
    clé web        AIOTECH_WEB_API_KEY_FILE : clé de l'interface, générée au premier démarrage
                   et injectée par nginx sur les seules routes de l'interface ; ses quotas
                   sont comptés par adresse IP du visiteur et elle n'ouvre jamais /admin ;
    administration AIOTECH_ADMIN_TOKEN, exigé EN PLUS d'une clé d'API valide sur toutes les
                   routes /admin (double contrôle) ; sans jeton configuré, l'administration
                   est fermée ;
    quotas         fenêtres fixes par clé : requêtes par minute et par jour, compteurs dans
                   Redis (ou en mémoire), le dépassement renvoie 429 avec Retry-After.
"""
from __future__ import annotations

import hashlib
import hmac
import time
from dataclasses import dataclass

from aiotech.storage.kv import KeyValueStore


def digest(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ApiPrincipal:
    name: str
    key_id: str


class AuthError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


class ApiKeyStore:
    def __init__(self, keys: dict[str, str], admin_token: str | None) -> None:
        self._keys = {digest(secret): name for name, secret in keys.items() if secret}
        self._admin = digest(admin_token) if admin_token else None

    @classmethod
    def from_env(cls, raw_keys: str, admin_token: str | None, extra: dict[str, str] | None = None) -> ApiKeyStore:
        keys: dict[str, str] = {}
        for name, secret in (extra or {}).items():
            if len(secret) < 16:
                raise ValueError(f"clé d'API « {name} » trop courte (16 caractères minimum)")
            keys[name] = secret
        for n, item in enumerate(part.strip() for part in raw_keys.split(",") if part.strip()):
            name, sep, secret = item.partition(":")
            if not sep:
                name, secret = f"cle-{n + 1}", item
            if len(secret) < 16:
                raise ValueError(f"clé d'API « {name} » trop courte (16 caractères minimum)")
            keys[name.strip()] = secret.strip()
        return cls(keys, admin_token.strip() if admin_token and admin_token.strip() else None)

    @property
    def enabled(self) -> bool:
        return bool(self._keys)

    @property
    def admin_enabled(self) -> bool:
        return self._admin is not None

    def authenticate(self, presented: str | None) -> ApiPrincipal:
        if not presented:
            raise AuthError(401, "Clé d'API manquante")
        candidate = digest(presented)
        match: str | None = None
        for known, name in self._keys.items():
            if hmac.compare_digest(candidate, known):
                match = name
        if match is None:
            raise AuthError(401, "Clé d'API invalide")
        return ApiPrincipal(name=match, key_id=candidate[:12])

    def authorize_admin(self, presented: str | None) -> None:
        if self._admin is None:
            raise AuthError(403, "Administration désactivée : AIOTECH_ADMIN_TOKEN non configuré")
        if not presented or not hmac.compare_digest(digest(presented), self._admin):
            raise AuthError(403, "Jeton d'administration invalide")


@dataclass(frozen=True)
class QuotaDecision:
    allowed: bool
    retry_after: int = 0
    minute_used: int = 0
    day_used: int = 0


class QuotaManager:
    def __init__(self, store: KeyValueStore, per_minute: int, per_day: int) -> None:
        self.store = store
        self.per_minute = per_minute
        self.per_day = per_day

    async def consume(self, key_id: str, now: float | None = None) -> QuotaDecision:
        t = time.time() if now is None else now
        minute_window, day_window = int(t // 60), int(t // 86400)
        minute = await self.store.incr(f"quota:{key_id}:m:{minute_window}", 1, ttl_seconds=120)
        day = await self.store.incr(f"quota:{key_id}:d:{day_window}", 1, ttl_seconds=172800)
        if self.per_minute > 0 and minute > self.per_minute:
            return QuotaDecision(False, retry_after=60 - int(t % 60), minute_used=minute, day_used=day)
        if self.per_day > 0 and day > self.per_day:
            return QuotaDecision(False, retry_after=86400 - int(t % 86400), minute_used=minute, day_used=day)
        return QuotaDecision(True, minute_used=minute, day_used=day)
