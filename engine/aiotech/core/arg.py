"""
ARG : Admissibility & Reachability Gate, recodé pour la recherche.

Le cœur aiotech v45 a remplacé la projection sphérique d'AIOTECH 44 par des critères à
unité explicite, tous dans [0, 1], combinés par la t-norme de Gödel :

    A(x) = T_G(c_1(x), ..., c_m(x)) = min_i c_i(x)          admissible  <=>  A(x) >= tau

Un seul critère défaillant suffit à rejeter : c'est la sémantique du « ET » logique.

Critères utilisés par AIOTECH Search :
    reach(q, x)    = 1 - ||relu(q - x)||_2 / ||q||_2
                     part de la masse euclidienne de la requête atteinte par x ;
    coverage(q, x) = somme des IDF des mots de q présents dans x / somme des IDF des mots de q ;
    constraint(x)  = degré de vérité des contraintes explicites (termes requis ou interdits).

Vérification de chaîne (synthèse) : chaque phrase doit être atteignable depuis les
preuves et les phrases précédentes ; le maillon le plus faible décide.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from aiotech.core.embeddings import HashingEmbedder, Matrix, Vector
from aiotech.core.text import content_words, normalize, split_sentences


def godel_tnorm(values: Iterable[float]) -> float:
    """T_G(a_1, ..., a_n) = min(a_1, ..., a_n) ; la conjonction vide vaut 1 (vrai)."""
    items = [float(v) for v in values]
    if not items:
        return 1.0
    return min(items)


def clamp01(value: float) -> float:
    if math.isnan(value):
        return 0.0
    return min(1.0, max(0.0, value))


def reach(q: Vector, x: Vector) -> float:
    """reach(q, x) dans [0, 1] : 1 si x contient toute la masse de q, 0 si aucune."""
    q_norm = float(np.linalg.norm(q))
    if q_norm == 0.0:
        return 1.0
    deficit = np.maximum(q - x, 0.0)
    return clamp01(1.0 - float(np.linalg.norm(deficit)) / q_norm)


def reach_many(q: Vector, xs: Matrix) -> list[float]:
    """reach(q, x_i) pour chaque ligne de xs, en un seul calcul vectoriel."""
    if xs.shape[0] == 0:
        return []
    q_norm = float(np.linalg.norm(q))
    if q_norm == 0.0:
        return [1.0] * xs.shape[0]
    deficits = np.linalg.norm(np.maximum(q[None, :] - xs, 0.0), axis=1)
    return [clamp01(1.0 - float(d) / q_norm) for d in deficits]


def idf_weights(query_words: Sequence[str], documents_words: Sequence[set[str]]) -> dict[str, float]:
    """IDF calculée sur les candidats : un mot présent partout ne discrimine rien."""
    n = len(documents_words)
    weights: dict[str, float] = {}
    for w in set(query_words):
        df = sum(1 for doc in documents_words if w in doc)
        weights[w] = math.log((n + 1.0) / (df + 0.5))
    return weights


def coverage(query_words: Sequence[str], segment_words: set[str], idf: dict[str, float]) -> float:
    unique = set(query_words)
    if not unique:
        return 1.0
    ordered = sorted(unique)  # ordre fixe : sommes flottantes identiques d'une exécution à l'autre
    total = sum(max(idf.get(w, 0.0), 0.0) for w in ordered)
    if total <= 0.0:
        return 1.0 if unique <= segment_words else len(unique & segment_words) / len(unique)
    return clamp01(sum(max(idf.get(w, 0.0), 0.0) for w in ordered if w in segment_words) / total)


@dataclass(frozen=True)
class ArgConstraints:
    """Contraintes lexicales vérifiables sur un texte.

    required_terms  : degré = fraction des termes présents ;
    forbidden_terms : un seul terme présent donne le degré 0 (rejet).
    """

    required_terms: tuple[str, ...] = ()
    forbidden_terms: tuple[str, ...] = ()

    def degree(self, text: str) -> float:
        norm = normalize(text)
        for term in self.forbidden_terms:
            if normalize(term) in norm:
                return 0.0
        if self.required_terms:
            present = sum(1 for t in self.required_terms if normalize(t) in norm)
            return present / len(self.required_terms)
        return 1.0


@dataclass(frozen=True)
class ChainReport:
    supports: tuple[float, ...]
    t_godel: float
    admissible: bool
    weakest_step: int | None
    steps: tuple[str, ...] = field(default_factory=tuple)

    @property
    def weakest_text(self) -> str | None:
        return self.steps[self.weakest_step] if self.weakest_step is not None else None


class ReachabilityGate:
    """Porte d'admissibilité : critères euclidiens et lexicaux, conjonction de Gödel."""

    def __init__(self, tau: float = 0.35, embedder: HashingEmbedder | None = None) -> None:
        if not 0.0 <= tau <= 1.0:
            raise ValueError("tau doit être dans [0, 1]")
        self.tau = tau
        # Pas de bigrammes : la requête est réduite à ses mots porteurs de sens.
        self.embedder = embedder or HashingEmbedder(bigram_weight=0.0)

    def query_vector(self, text: str) -> Vector:
        meaningful = content_words(text)
        return self.embedder.embed_one(" ".join(meaningful) if meaningful else text)

    def admissibility(
        self,
        query: str,
        segments: Sequence[str],
        constraints: ArgConstraints | None = None,
    ) -> list[tuple[float, float, float, float]]:
        """(reach, coverage, constraint, admissibility) pour chaque segment."""
        if not segments:
            return []
        cons = constraints or ArgConstraints()
        q_words = content_words(query)
        q_vec = self.query_vector(query)
        reaches = reach_many(q_vec, self.embedder.embed(list(segments)))
        seg_words = [set(content_words(s)) for s in segments]
        idf = idf_weights(q_words, seg_words)
        rows: list[tuple[float, float, float, float]] = []
        for i, seg in enumerate(segments):
            cov = coverage(q_words, seg_words[i], idf)
            con = cons.degree(seg)
            rows.append((reaches[i], cov, con, godel_tnorm((reaches[i], cov, con))))
        return rows

    def verify_chain(
        self,
        steps: Sequence[str] | str,
        context: Sequence[str] = (),
        query: str = "",
        tau: float | None = None,
    ) -> ChainReport:
        """Chaque étape doit être atteignable depuis le contexte, la requête et les étapes précédentes."""
        items = split_sentences(steps) if isinstance(steps, str) else [s for s in steps if s.strip()]
        threshold = self.tau if tau is None else tau
        if not items:
            return ChainReport((), 1.0, True, None, ())
        known = np.zeros(self.embedder.dim, dtype=np.float32)
        for source in [*context, query] if query else list(context):
            known = np.maximum(known, self.embedder.embed_one(source))
        supports: list[float] = []
        for step in items:
            meaningful = content_words(step)
            if not meaningful:
                supports.append(1.0)
            else:
                supports.append(reach(self.embedder.embed_one(" ".join(meaningful)), known))
            known = np.maximum(known, self.embedder.embed_one(step))
        t = godel_tnorm(supports)
        weakest = int(np.argmin(np.asarray(supports)))
        return ChainReport(tuple(supports), t, t >= threshold, weakest, tuple(items))
