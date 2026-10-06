"""
Source web : SearXNG (méta-moteur libre, auto-hébergé dans Docker, sans clé d'API).

Seule l'API JSON de l'instance configurée est appelée : les pages de résultats ne sont
jamais téléchargées par le moteur (pas de requête vers une URL fournie par un tiers,
donc pas de SSRF). Le texte exploité est l'extrait (« content ») renvoyé par SearXNG.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass

import httpx

from aiotech.models import Document

log = logging.getLogger("aiotech.web")


@dataclass
class SearxngClient:
    base_url: str
    http: httpx.AsyncClient
    max_results: int = 6
    language: str = "fr"
    safesearch: int = 1

    async def search(self, query: str) -> list[Document]:
        params: dict[str, str | int] = {
            "q": query, "format": "json", "language": self.language, "safesearch": self.safesearch,
        }
        response = await self.http.get(f"{self.base_url.rstrip('/')}/search", params=params)
        response.raise_for_status()
        payload = response.json()
        results = payload.get("results", []) if isinstance(payload, dict) else []
        documents: list[Document] = []
        seen: set[str] = set()
        for item in results:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "")
            content = str(item.get("content") or "").strip()
            title = str(item.get("title") or url).strip()
            if not url.startswith(("http://", "https://")) or not content or url in seen:
                continue
            seen.add(url)
            digest = hashlib.blake2b(url.encode(), digest_size=8).hexdigest()
            documents.append(
                Document(id=f"w_{digest}", title=title[:200], text=content[:2000], url=url, origin="web")
            )
            if len(documents) >= self.max_results:
                break
        return documents
