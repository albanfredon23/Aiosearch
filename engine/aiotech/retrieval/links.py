"""
Liens entre documents par mention de titre : un passage qui cite le titre d'un autre document
(« … réalisé par Scott Derrickson … ») est relié à ce document, comme par un lien hypertexte.

C'est l'arête document -> document du graphe de recherche : elle sert au saut par liens du
pipeline, qui retrouve la seconde pièce d'une question en chaîne (le réalisateur d'un film, puis
sa nationalité) même quand la question ne la nomme pas. La comparaison se fait sur les mots
normalisés (sans casse ni accents) ; une précision finale entre parenthèses est ignorée
(« Ed Wood (film) » est cité comme « Ed Wood »).
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from aiotech.core.text import content_words, words
from aiotech.models import Document

_TRAILING_QUALIFIER = re.compile(r"\s*\([^()]*\)\s*$")
MIN_TITLE_CHARS = 3


def title_key(title: str) -> tuple[str, ...]:
    """Mots normalisés du titre, sans précision finale ; vide si le titre ne porte aucun mot de sens."""
    core = _TRAILING_QUALIFIER.sub("", title).strip() or title
    key = tuple(words(core))
    if not content_words(core) or len("".join(key)) < MIN_TITLE_CHARS:
        return ()
    return key


@dataclass
class TitleLinks:
    _by_first_word: dict[str, list[tuple[tuple[str, ...], str]]] = field(default_factory=dict)

    def build(self, documents: Iterable[Document]) -> None:
        index: dict[str, list[tuple[tuple[str, ...], str]]] = {}
        for document in documents:
            key = title_key(document.title)
            if key:
                index.setdefault(key[0], []).append((key, document.id))
        self._by_first_word = index

    def mentioned(self, text: str) -> list[str]:
        """Documents dont le titre apparaît dans le texte, dans l'ordre de première mention."""
        tokens = words(text)
        found: list[str] = []
        for start, token in enumerate(tokens):
            for key, document_id in self._by_first_word.get(token, ()):
                if document_id not in found and tuple(tokens[start:start + len(key)]) == key:
                    found.append(document_id)
        return found
