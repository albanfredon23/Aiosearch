"""
Échantillons figés versionnés dans le paquet (voir aiotech/bench/prepare.py et data/SOURCES.md).
"""
from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from importlib import resources
from typing import Any

from aiotech.models import Document

HOTPOT_FILE = "hotpotqa_dev_distractor_s300.jsonl.gz"
FEVER_FILE = "fever_dev_pairs.jsonl.gz"


def _rows(name: str) -> list[dict[str, Any]]:
    raw = resources.files("aiotech.bench.data").joinpath(name).read_bytes()
    return [json.loads(line) for line in gzip.decompress(raw).decode("utf-8").splitlines() if line.strip()]


@dataclass(frozen=True)
class HotpotItem:
    id: str
    question: str
    answer: str
    type: str
    supporting_titles: frozenset[str]
    paragraphs: tuple[tuple[str, str], ...]

    @property
    def yes_no(self) -> bool:
        return self.answer.strip().lower() in {"yes", "no"}

    def documents(self) -> list[Document]:
        """Un document par paragraphe ; le titre précède le texte (indexé de la même façon pour tous)."""
        return [Document(id=f"p{i}", title=title, text=f"{title}. {text}", reliability=0.7)
                for i, (title, text) in enumerate(self.paragraphs)]


def load_hotpot(limit: int | None = None) -> list[HotpotItem]:
    items = []
    for row in _rows(HOTPOT_FILE):
        paragraphs = tuple((p["title"], " ".join(s.strip() for s in p["sentences"]).strip()) for p in row["context"])
        items.append(HotpotItem(id=row["id"], question=row["question"], answer=row["answer"], type=row["type"],
                                supporting_titles=frozenset(row["supporting_titles"]), paragraphs=paragraphs))
    return items[:limit] if limit else items


@dataclass(frozen=True)
class FeverClaim:
    id: str
    claim: str
    label: str
    evidence_id: str


@dataclass(frozen=True)
class FeverSet:
    claims: tuple[FeverClaim, ...]
    evidence: dict[str, str]
    synthetic: frozenset[str]

    def documents(self) -> list[Document]:
        """Corpus commun : toutes les phrases de preuve (originales et leurres modifiés)."""
        return [Document(id=eid, title=eid, text=text, reliability=0.7) for eid, text in sorted(self.evidence.items())]


def _detokenize(text: str) -> str:
    """Les phrases FEVER sont tokenisées (« -LRB- », « , ») ; on rétablit la typographie courante."""
    for raw, char in (("-LRB-", "("), ("-RRB-", ")"), ("-LSB-", "["), ("-RSB-", "]"), ("-LCB-", "{"), ("-RCB-", "}"),
                      ("--", "–"), ("``", '"'), ("''", '"')):
        text = text.replace(raw, char)
    for punct in (" ,", " .", " ;", " :", " ?", " !", " 's", " 'd", " n't", " )", " ]"):
        text = text.replace(punct, punct[1:])
    return text.replace("( ", "(").replace("[ ", "[").strip()


def load_fever(limit: int | None = None) -> FeverSet:
    claims: list[FeverClaim] = []
    evidence: dict[str, str] = {}
    synthetic: set[str] = set()
    for row in _rows(FEVER_FILE):
        if row["kind"] == "claim":
            claims.append(FeverClaim(id=row["id"], claim=_detokenize(row["claim"]), label=row["label"],
                                     evidence_id=row["evidence_id"]))
        else:
            evidence[row["id"]] = _detokenize(row["text"])
            if row["synthetic"]:
                synthetic.add(row["id"])
    return FeverSet(claims=tuple(claims[:limit] if limit else claims), evidence=evidence,
                    synthetic=frozenset(synthetic))
