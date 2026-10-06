"""
Corpus documentaire : documents, passages, affirmations pré-extraites, index hybride.

Les affirmations par règles sont extraites UNE fois à l'ingestion et gardées avec chaque
passage : une recherche ne recalcule rien pour le corpus (frugalité, latence stable).
Persistance optionnelle : un fichier JSON par document dans `persist_dir`, écrit de
façon atomique (fichier temporaire puis renommage).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from aiotech.core.text import chunk_text
from aiotech.graph.claims import RuleExtractor
from aiotech.models import Claim, Document, Passage, ScoredPassage, Source
from aiotech.retrieval.index import PassageIndex
from aiotech.retrieval.reliability import ReliabilityModel

_SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")


def document_id_for(title: str, text: str) -> str:
    digest = hashlib.blake2b(f"{title}\n{text}".encode(), digest_size=8).hexdigest()
    return f"d_{digest}"


def source_for(document: Document, reliability: ReliabilityModel) -> Source:
    return Source(
        id=document.id,
        title=document.title,
        url=document.url,
        origin=document.origin,
        reliability=reliability.score(document.url, document.reliability, document.origin),
    )


def passages_for(document: Document, source: Source, chunk_tokens: int) -> list[Passage]:
    chunks = chunk_text(document.text, chunk_tokens) or [document.text]
    return [
        Passage(id=f"{document.id}#{i}", document_id=document.id, text=chunk, source=source)
        for i, chunk in enumerate(chunks)
        if chunk.strip()
    ]


def extract_document_claims(passages: Iterable[Passage], extractor: RuleExtractor) -> dict[str, list[Claim]]:
    """Affirmations par passage ; l'entité en cours passe d'un passage au suivant."""
    focus: str | None = None
    out: dict[str, list[Claim]] = {}
    for passage in passages:
        claims, focus = extractor.extract(passage, focus)
        out[passage.id] = claims
    return out


@dataclass
class CorpusStore:
    reliability: ReliabilityModel = field(default_factory=ReliabilityModel)
    chunk_tokens: int = 120
    persist_dir: Path | None = None
    extractor: RuleExtractor = field(default_factory=RuleExtractor)
    documents: dict[str, Document] = field(default_factory=dict)
    passages: dict[str, Passage] = field(default_factory=dict)
    claims: dict[str, list[Claim]] = field(default_factory=dict)
    version: int = 0
    _index: PassageIndex = field(default_factory=PassageIndex)
    _dirty: bool = True
    _lock: threading.RLock = field(default_factory=threading.RLock)

    def add(self, document: Document, persist: bool = True) -> list[Passage]:
        if not _SAFE_ID.match(document.id):
            raise ValueError("identifiant de document invalide (lettres, chiffres, _ . - ; 80 caractères max)")
        with self._lock:
            if document.id in self.documents:
                self._remove_unlocked(document.id)
            source = source_for(document, self.reliability)
            passages = passages_for(document, source, self.chunk_tokens)
            self.documents[document.id] = document
            for p in passages:
                self.passages[p.id] = p
            self.claims.update(extract_document_claims(passages, self.extractor))
            self.version += 1
            self._dirty = True
            if persist and self.persist_dir is not None:
                self._write(document)
            return passages

    def add_many(self, documents: Iterable[Document], persist: bool = True) -> int:
        count = 0
        for doc in documents:
            self.add(doc, persist=persist)
            count += 1
        return count

    def remove(self, document_id: str) -> bool:
        with self._lock:
            if document_id not in self.documents:
                return False
            self._remove_unlocked(document_id)
            self.version += 1
            self._dirty = True
            if self.persist_dir is not None:
                path = self.persist_dir / f"{document_id}.json"
                if path.exists():
                    path.unlink()
            return True

    def _remove_unlocked(self, document_id: str) -> None:
        del self.documents[document_id]
        for pid in [pid for pid, p in self.passages.items() if p.document_id == document_id]:
            del self.passages[pid]
            self.claims.pop(pid, None)

    def search(self, query: str, top_k: int) -> list[ScoredPassage]:
        with self._lock:
            if self._dirty:
                self._index.build(list(self.passages.values()))
                self._dirty = False
            return self._index.search(query, top_k)

    def claims_for(self, passage_id: str) -> list[Claim]:
        return list(self.claims.get(passage_id, []))

    def stats(self) -> dict[str, int]:
        return {
            "documents": len(self.documents),
            "passages": len(self.passages),
            "claims": sum(len(c) for c in self.claims.values()),
            "version": self.version,
        }

    def _write(self, document: Document) -> None:
        assert self.persist_dir is not None
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        target = self.persist_dir / f"{document.id}.json"
        fd, tmp = tempfile.mkstemp(dir=self.persist_dir, prefix=".tmp-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(document.model_dump(), handle, ensure_ascii=False, indent=2)
            os.replace(tmp, target)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    def load_dir(self, directory: Path) -> int:
        if not directory.is_dir():
            return 0
        count = 0
        for path in sorted(directory.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            items = payload if isinstance(payload, list) else [payload]
            for item in items:
                self.add(Document.model_validate(item), persist=False)
                count += 1
        return count
