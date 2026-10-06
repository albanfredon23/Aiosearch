"""
Entités : nettoyage des sujets, clés normalisées et résolution des alias.

Deux mentions désignent la même entité si leurs mots porteurs de sens se recouvrent
assez ET si leurs identifiants chiffrés sont identiques : « IdeaPad Slim 3 » et
« Lenovo IdeaPad Slim 3 » fusionnent, « IdeaPad Slim 3 » et « IdeaPad Slim 5 » jamais
(garde « identifiants » reprise du cache aiotech v45).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from aiotech.core.text import STOPWORDS, content_words, identifiers, jaccard, normalize, stem, words

_LEADING_DETERMINERS = re.compile(
    r"^(?:(?:le|la|les|un|une|des|du|de la|ce|cet|cette|ces|son|sa|ses|leur|leurs|notre|votre|"
    r"the|a|an|this|that|these|those|its|their|our|your|selon|according to)\s+|(?:l|d|de l)['’]\s*)+",
    re.IGNORECASE,
)
_ATTRIBUTE_LEADS = re.compile(
    r"^(?:prix|tarif|coût|cout|price|cost|poids|weight|capacité|capacite|mémoire vive|memoire vive|ram|"
    r"stockage|storage)\s+(?:de la |de l'|du |des |de |d'|of the |of )",
    re.IGNORECASE,
)
PRONOUN_SUBJECTS = frozenset(
    {
        "il", "elle", "ils", "elles", "it", "they", "celui-ci", "celle-ci", "ceux-ci", "ce dernier",
        "cette derniere", "ce modele", "cet ordinateur", "cette machine", "ce pc", "cette configuration",
        "ce produit", "this model", "this computer", "this machine", "the latter", "ce portable",
        "cette tour", "on", "nous", "vous", "ce", "cela", "ceci", "this", "that",
    }
)
_ATTRIBUTE_NOUNS = frozenset(
    {
        "prix", "tarif", "cout", "price", "cost", "poids", "weight", "capacite", "memoire vive", "ram",
        "stockage", "storage", "ssd", "processeur", "processor", "ecran", "screen", "autonomie", "budget",
        "garantie", "warranty", "memoire", "memory",
    }
)
_MAX_SUBJECT_WORDS = 12


def clean_subject(raw: str) -> str | None:
    """Sujet lisible ou None si c'est un pronom, une phrase trop longue ou du vide."""
    text = raw.strip().strip(" ,;:-–—\"'«»()")
    if "," in text:
        text = text.rsplit(",", 1)[1].strip()
    text = _LEADING_DETERMINERS.sub("", text).strip()
    text = _ATTRIBUTE_LEADS.sub("", text).strip()
    text = _LEADING_DETERMINERS.sub("", text).strip()
    if not text:
        return None
    if normalize(text) in PRONOUN_SUBJECTS or normalize(text) in _ATTRIBUTE_NOUNS:
        return None
    if len(text.split()) > _MAX_SUBJECT_WORDS:
        return None
    if not content_words(text):
        return None
    return text


def entity_key(text: str) -> str:
    """Mots porteurs de sens racinisés ; les numéros de modèle (« 3 », « 44 ») sont gardés."""
    keep: list[str] = []
    for w in words(text):
        if any(ch.isdigit() for ch in w):
            keep.append(w)
        elif w not in STOPWORDS and len(w) > 1:
            keep.append(stem(w))
    return " ".join(keep)


def same_entity(key_a: str, key_b: str) -> bool:
    if key_a == key_b:
        return True
    if identifiers(key_a) != identifiers(key_b):
        return False
    a, b = set(key_a.split()), set(key_b.split())
    if not a or not b:
        return False
    if (a <= b or b <= a) and min(len(a), len(b)) >= 2:
        return True
    return jaccard(a, b) >= 0.6


@dataclass
class EntityResolver:
    """Associe chaque clé à une clé canonique (la première vue) et garde le meilleur libellé."""

    canonical: dict[str, str] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)

    def resolve(self, key: str, label: str) -> str:
        if key in self.canonical:
            return self.canonical[key]
        for known in list(self.labels):
            if same_entity(key, known):
                self.canonical[key] = known
                if len(label) > len(self.labels[known]):
                    self.labels[known] = label
                return known
        self.canonical[key] = key
        self.labels[key] = label
        return key

    def label(self, key: str) -> str:
        return self.labels.get(self.canonical.get(key, key), key)
