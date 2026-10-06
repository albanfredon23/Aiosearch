"""
Fiabilité documentaire a priori d'une source, dans [0, 1].

Ordre de priorité : fiabilité déclarée sur le document (corpus maîtrisé) > règle de
domaine (sources officielles, scientifiques, encyclopédiques, forums) > valeur par défaut
selon l'origine. Les règles se complètent ou se remplacent par AIOTECH_SOURCE_RELIABILITY
(JSON {"motif regex de domaine": score}).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

from aiotech.models import Origin

DEFAULT_DOMAIN_RULES: tuple[tuple[str, float], ...] = (
    (r"(^|\.)legifrance\.gouv\.fr$", 0.97),
    (r"(^|\.)service-public\.(fr|gouv\.fr)$", 0.95),
    (r"(^|\.)gouv\.fr$", 0.95),
    (r"(^|\.)europa\.eu$", 0.95),
    (r"(^|\.)who\.int$", 0.93),
    (r"(^|\.)has-sante\.fr$", 0.93),
    (r"(^|\.)(inserm|cnrs|inria|insee)\.fr$", 0.9),
    (r"(^|\.)(nature|science|thelancet|nejm|bmj)\.(com|org)$", 0.9),
    (r"(^|\.)ncbi\.nlm\.nih\.gov$", 0.9),
    (r"(^|\.)(edu|ac\.[a-z]{2})$", 0.85),
    (r"(^|\.)arxiv\.org$", 0.75),
    (r"(^|\.)wikipedia\.org$", 0.75),
    (r"(^|\.)(reddit|quora|jeuxvideo|doctissimo)\.(com|fr)$", 0.35),
    (r"forum", 0.35),
    (r"(^|\.)medium\.com$|blog", 0.45),
)


def domain_of(url: str | None) -> str:
    if not url:
        return ""
    try:
        host = urlparse(url).hostname or ""
    except ValueError:
        return ""
    return host.lower().removeprefix("www.")


@dataclass
class ReliabilityModel:
    corpus_default: float = 0.7
    web_default: float = 0.55
    rules: list[tuple[re.Pattern[str], float]] = field(
        default_factory=lambda: [(re.compile(p), s) for p, s in DEFAULT_DOMAIN_RULES]
    )

    @classmethod
    def from_json(cls, raw: str | None) -> ReliabilityModel:
        model = cls()
        if not raw:
            return model
        extra = json.loads(raw)
        if not isinstance(extra, dict):
            raise TypeError("AIOTECH_SOURCE_RELIABILITY doit être un objet JSON {motif: score}")
        custom = [(re.compile(str(p)), min(1.0, max(0.0, float(s)))) for p, s in extra.items()]
        model.rules = custom + model.rules
        return model

    def score(self, url: str | None, declared: float | None, origin: Origin) -> float:
        if declared is not None:
            return declared
        domain = domain_of(url)
        if domain:
            for pattern, value in self.rules:
                if pattern.search(domain):
                    return value
        return self.web_default if origin == "web" else self.corpus_default
