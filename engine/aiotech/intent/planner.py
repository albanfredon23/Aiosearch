"""
Intent & Query Engine (planificateur par règles, sans appel LLM).

La requête n'est pas traitée littéralement : elle est décomposée en un objectif, un type
de question, des contraintes explicites, des entités, puis en plusieurs interprétations
(les « neurones » de l'interface), chacune portant ses propres sous-requêtes.
Exemple : « Quel ordinateur à 500 € serait le meilleur pour faire tourner AIOTECH 44 ? »
    contraintes      prix ≤ 500 €
    interprétations  T1 PC portable, T2 PC fixe, T3 occasion ou reconditionné

Ce planificateur sert seul hors ligne et en repli du planificateur LLM (llm/planner.py),
qui sait en plus distinguer les sens d'un mot ambigu.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from importlib import resources

from aiotech.core.text import STOPWORDS, content_words, has_phrase, normalize, words
from aiotech.core.units import unit_spec
from aiotech.graph.entities import clean_subject
from aiotech.intent.constraints import extract_constraints
from aiotech.models import IntentPlan, Interpretation, QuestionType

_RECOMMENDATION_RE = re.compile(
    r"\b(?:meilleure?s?|best|recommand\w*|conseill\w*|choisir|choose|lequel|laquelle|lesquel\w*|"
    r"top\s+\d+|id[ée]ale?s?|which\s+\w+\s+should|should\s+i\s+buy|que\s+prendre|quoi\s+acheter)\b",
    re.IGNORECASE,
)
_COMPARISON_RE = re.compile(r"\b(?:vs\.?|versus|compar\w*|diff[ée]rences?\s+entre|difference\s+between)\b", re.IGNORECASE)
_DEFINITION_RE = re.compile(
    r"(?:qu['’]est[- ]ce\s+qu['’]?|c['’]est\s+quoi|what\s+(?:is|are)\s+(?:a|an|the)?|d[ée]finition|"
    r"que\s+signifie|what\s+does\s+\w+\s+mean|explique[rz]?)",
    re.IGNORECASE,
)
_OFFICIAL_RE = re.compile(
    r"\b(?:loi|l[ée]gal\w*|droit|d[ée]lai|obligation\w*|r[èe]gle\w*|norme\w*|r[ée]glement\w*|contrat|"
    r"imp[ôo]t\w*|dose\w*|m[ée]dicament\w*|sant[ée]|law|legal|regulation|tax|dosage)\b",
    re.IGNORECASE,
)
_QUESTION_LEAD_RE = re.compile(
    r"^\s*(?:quel(?:le)?s?\s+(?:est|sont)\s+|qu['’]est[- ]ce\s+(?:que\s+|qu['’])|c['’]est\s+quoi\s+|comment\s+|"
    r"combien\s+(?:de\s+)?|pourquoi\s+|o[uù]\s+|quand\s+|qui\s+|what\s+(?:is|are)\s+|how\s+(?:much|many|to)?\s*|"
    r"why\s+|where\s+|when\s+|which\s+|who\s+)",
    re.IGNORECASE,
)
_PHRASE_SPLIT_RE = re.compile(r"\s+(?:pour|sur|avec|dans|chez|et|ou|for|with|about|and|or)\s+|[,;?!]", re.IGNORECASE)
_ENTITY_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9\-]+|[a-z]+[A-Z][A-Za-z0-9\-]*)(?:\s+(?:[A-Z][A-Za-z0-9\-]*|[a-z]+[A-Z][A-Za-z0-9\-]*|\d+[A-Za-z]*))*"
)
_ENTITY_STOP = frozenset(
    {"quel", "quelle", "quels", "quelles", "le", "la", "les", "un", "une", "what", "which", "is", "est", "comment",
     "pourquoi", "combien", "qui", "ou", "how", "why", "the", "a", "an", "pc", "je", "i"}
)
_TECH_TOKENS = frozenset({"ram", "ssd", "hdd", "cpu", "gpu", "pc", "os", "usb", "wifi", "ia", "ai", "api", "faq", "tva"})
_COMPARISON_SPLIT_RE = re.compile(r"\s+(?:vs\.?|versus|ou|or|contre|against)\s+", re.IGNORECASE)


@dataclass(frozen=True)
class Facet:
    facet: str
    label: str
    terms: tuple[str, ...]
    expansion: str


@dataclass(frozen=True)
class FacetGroup:
    triggers: frozenset[str]
    facets: tuple[Facet, ...]


def load_facets(raw: str | None = None) -> list[FacetGroup]:
    text = raw if raw is not None else resources.files("aiotech.data").joinpath("facets.json").read_text("utf-8")
    groups: list[FacetGroup] = []
    for item in json.loads(text):
        groups.append(
            FacetGroup(
                triggers=frozenset(normalize(t) for t in item["triggers"]),
                facets=tuple(
                    Facet(f["facet"], f["label"], tuple(f["terms"]), f["expansion"]) for f in item["facets"]
                ),
            )
        )
    return groups


def detect_question_type(query: str) -> QuestionType:
    if _COMPARISON_RE.search(query):
        return "comparison"
    if _RECOMMENDATION_RE.search(query):
        return "recommendation"
    if _DEFINITION_RE.search(query):
        return "definition"
    return "factual"


def query_entities(query: str) -> list[str]:
    found: list[str] = []
    for match in _ENTITY_RE.finditer(query):
        tokens = match.group(0).split()
        while tokens and normalize(tokens[0]) in _ENTITY_STOP:
            tokens = tokens[1:]
        if len(tokens) == 1 and (normalize(tokens[0]) in _TECH_TOKENS or unit_spec(tokens[0]) is not None):
            continue
        if not tokens:
            continue
        entity = " ".join(tokens)
        if entity not in found and (len(tokens) > 1 or any(ch.isdigit() for ch in entity) or entity.isupper()
                                    or match.start() > 0):
            found.append(entity)
    return found


def key_phrases(query: str) -> list[str]:
    body = _QUESTION_LEAD_RE.sub("", query.strip())
    phrases: list[str] = []
    for part in _PHRASE_SPLIT_RE.split(body):
        cleaned = clean_subject(part or "")
        if cleaned and cleaned not in phrases:
            phrases.append(cleaned)
    return phrases


def _core_terms(query: str, exclude: frozenset[str]) -> str:
    keep = [w for w in words(query) if w not in STOPWORDS and len(w) > 1 and w not in exclude]
    return " ".join(keep)


class RulePlanner:
    def __init__(self, facets: list[FacetGroup] | None = None) -> None:
        self.facets = facets if facets is not None else load_facets()

    def plan(self, query: str, max_interpretations: int = 4) -> IntentPlan:
        question_type = detect_question_type(query)
        constraints = tuple(extract_constraints(query))
        entities = tuple(query_entities(query))
        key_terms = tuple(dict.fromkeys(content_words(query)))
        interpretations = self._interpretations(query, question_type, entities, max(2, max_interpretations))
        objective = _QUESTION_LEAD_RE.sub("", query.strip()).rstrip(" ?!.") or query.strip()
        return IntentPlan(
            query=query,
            objective=objective,
            question_type=question_type,
            constraints=constraints,
            key_terms=key_terms,
            entities=entities,
            interpretations=tuple(interpretations[:max_interpretations]),
            planner="rules",
        )

    def _interpretations(
        self, query: str, question_type: QuestionType, entities: tuple[str, ...], limit: int
    ) -> list[Interpretation]:
        if question_type == "comparison":
            items = [i for i in _COMPARISON_SPLIT_RE.split(_QUESTION_LEAD_RE.sub("", query).strip(" ?!.")) if i.strip()]
            if len(items) >= 2:
                return self._comparison_interpretations(query, items, limit)
        query_words = set(words(query))
        group = next((g for g in self.facets if g.triggers & query_words), None)
        if group is not None:
            return self._facet_interpretations(query, group, limit)
        return self._generic_interpretations(query, question_type, limit)

    @staticmethod
    def _facet_interpretations(query: str, group: FacetGroup, limit: int) -> list[Interpretation]:
        explicit = [f for f in group.facets if any(has_phrase(query, t) for t in f.terms)]
        exclude = frozenset(group.triggers) | frozenset(normalize(t) for f in group.facets for t in f.terms)
        core = _core_terms(query, exclude)
        out: list[Interpretation] = []
        for rank, facet in enumerate(group.facets):
            plausibility = 0.9 if facet in explicit else (0.35 if explicit else round(0.6 - 0.02 * rank, 2))
            subqueries = tuple(s for s in (f"{facet.expansion} {core}".strip(), facet.expansion) if s)
            out.append(
                Interpretation(
                    id="",
                    label=facet.label,
                    facet=facet.facet,
                    rationale=f"Lecture « {facet.label} » de la demande",
                    subqueries=tuple(dict.fromkeys(subqueries)),
                    facet_terms=facet.terms,
                    plausibility=plausibility,
                )
            )
        out.sort(key=lambda i: i.plausibility, reverse=True)
        return [i.model_copy(update={"id": f"T{n}"}) for n, i in enumerate(out[:limit], start=1)]

    @staticmethod
    def _comparison_interpretations(query: str, items: list[str], limit: int) -> list[Interpretation]:
        out = [
            Interpretation(
                id="", label="Comparaison directe", facet="comparaison",
                rationale="Les éléments comparés côte à côte", subqueries=(query,), plausibility=0.7,
            )
        ]
        for item in items:
            label = clean_subject(item) or item.strip()
            out.append(
                Interpretation(
                    id="", label=label, facet=normalize(label), rationale=f"Ce que les sources disent de {label}",
                    subqueries=(label,), facet_terms=(label,), plausibility=0.6,
                )
            )
        return [i.model_copy(update={"id": f"T{n}"}) for n, i in enumerate(out[:limit], start=1)]

    @staticmethod
    def _generic_interpretations(query: str, question_type: QuestionType, limit: int) -> list[Interpretation]:
        phrases = key_phrases(query)
        main = phrases[0] if phrases else query
        out = [
            Interpretation(
                id="", label="Lecture littérale", facet="litteral",
                rationale="La question telle qu'elle est posée", subqueries=(query,), plausibility=0.7,
            )
        ]
        if question_type == "definition":
            out.append(
                Interpretation(
                    id="", label=f"Définition de « {main} »", facet="definition",
                    rationale="Ce que désigne la notion", subqueries=(f"{main} définition", main), plausibility=0.65,
                )
            )
        for phrase in phrases[:2]:
            out.append(
                Interpretation(
                    id="", label=f"Centré sur « {phrase} »", facet=normalize(phrase),
                    rationale=f"La question vue depuis « {phrase} »", subqueries=(phrase,), plausibility=0.55,
                )
            )
        if _OFFICIAL_RE.search(query):
            out.append(
                Interpretation(
                    id="", label="Cadre officiel", facet="officiel",
                    rationale="Ce que disent les textes et sources officielles",
                    subqueries=(f"{main} loi officiel", main), plausibility=0.5,
                )
            )
        unique: dict[tuple[str, ...], Interpretation] = {}
        for interp in out:
            unique.setdefault(interp.subqueries, interp)
        ordered = sorted(unique.values(), key=lambda i: i.plausibility, reverse=True)
        return [i.model_copy(update={"id": f"T{n}"}) for n, i in enumerate(ordered[:limit], start=1)]
