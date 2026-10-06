"""
Outils texte du cœur AIOTECH Search : normalisation, mots, racinisation légère FR/EN,
estimation de tokens, découpage en phrases et en segments.

Repris du cœur aiotech v45 (même racinisation, même liste de mots vides) pour que les
scores de l'ARG restent comparables entre les deux projets.
"""
from __future__ import annotations

import math
import re
import unicodedata

_WORD_RE = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)?")
_SENTENCE_RE = re.compile(r"(?<=[.!?;])\s+|\n+")

STOPWORDS: frozenset[str] = frozenset(
    """
    a au aux avec ce ces cet cette dans de des du elle en et est etre il ils je la le les leur lui
    ma mais me meme mes moi mon ne nos notre nous on ou par pas pour qu que quel quelle quels quelles
    qui sa se ses son sont sur ta te tes toi ton tu un une vos votre vous y d l s n c j m t
    comment combien quoi ou quand pourquoi sous peut peuvent lequel laquelle lesquels lesquelles
    faire fait dire avoir plus moins tres aussi entre ainsi alors donc cela ceci ca chaque quelque
    the a an and are as at be by for from has have in is it its of on or that the to was were what
    which who whom why how with this these those do does did can could should would will about
    into than then there their them they also more less very
    """.split()
)

_SUFFIXES: tuple[str, ...] = (
    "issements", "issement", "atrices", "ateurs", "ations", "ements", "ement", "atrice", "ateur",
    "ation", "ances", "ences", "ance", "ence", "euses", "euse", "ites", "ite", "ives", "ive",
    "ees", "ers", "ent", "ee", "er", "ez", "es", "e", "s", "x",
)


def stem(word: str) -> str:
    """Racinisation légère : un seul suffixe retiré, racine d'au moins 4 lettres.

    Les jetons contenant un chiffre (références, montants) sont gardés tels quels.
    """
    if any(ch.isdigit() for ch in word):
        return word
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def normalize(text: str) -> str:
    """Minuscules, sans accents."""
    return strip_accents(text.lower())


def words(text: str) -> list[str]:
    """Mots normalisés ; les nombres décimaux restent d'un seul tenant."""
    return _WORD_RE.findall(normalize(text))


def content_words(text: str) -> list[str]:
    """Mots porteurs de sens, racinisés (mots vides et lettres isolées retirés)."""
    return [stem(w) for w in words(text) if w not in STOPWORDS and len(w) > 1]


def identifiers(text: str) -> frozenset[str]:
    """Jetons contenant un chiffre : références, montants, codes produit."""
    return frozenset(w for w in words(text) if any(ch.isdigit() for ch in w))


def estimate_tokens(text: str) -> int:
    """Estimation usuelle des tokeniseurs BPE : environ 4 caractères par token.

    L'usage exact renvoyé par le fournisseur LLM est toujours préféré quand il existe.
    """
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def split_sentences(text: str) -> list[str]:
    """Découpe en phrases sans couper les nombres décimaux (« 4,5 », « 1.5 »)."""
    parts = _SENTENCE_RE.split(text.strip())
    return [p.strip() for p in parts if p and p.strip()]


def chunk_text(text: str, max_tokens: int = 120) -> list[str]:
    """Segments d'environ `max_tokens`, sans couper de phrase.

    Des segments de taille comparable gardent les distances euclidiennes de l'ARG comparables.
    """
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for sentence in split_sentences(text):
        tokens = estimate_tokens(sentence)
        if current and current_tokens + tokens > max_tokens:
            chunks.append(" ".join(current))
            current, current_tokens = [], 0
        current.append(sentence)
        current_tokens += tokens
    if current:
        chunks.append(" ".join(current))
    return chunks


def jaccard(a: set[str] | frozenset[str], b: set[str] | frozenset[str]) -> float:
    """Indice de Jaccard sur deux ensembles de mots (1.0 si les deux sont vides)."""
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def has_phrase(text: str, phrase: str) -> bool:
    """`phrase` apparaît dans `text` comme suite de mots entiers, au pluriel près
    (« exception » trouve « exceptions », « tour » ne trouve pas « tourner »)."""
    target = [stem(w) for w in words(phrase)]
    if not target:
        return False
    return f" {' '.join(target)} " in f" {' '.join(stem(w) for w in words(text))} "
