"""
Extraction d'affirmations par règles (FR/EN), sans appel LLM.

Chaque phrase d'un passage est lue une fois : les quantités (prix, mémoire, durées...)
deviennent des affirmations « entité -attribut-> valeur » avec leur citation exacte,
les phrases « X est un Y » donnent le type de l'entité, et les phrases d'exigence
(« X nécessite au moins 16 Go de RAM ») deviennent des contraintes documentées.

Les pronoms (« Il dispose de... ») sont rattachés à l'entité en cours du document.
L'extraction LLM (llm/extraction.py) complète ces règles quand un modèle est configuré ;
ses affirmations ne sont gardées que si leur citation figure mot pour mot dans la source.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from aiotech.core.text import split_sentences
from aiotech.core.units import iter_quantities
from aiotech.graph.entities import clean_subject, entity_key
from aiotech.models import Claim, ClaimKind, Comparator, Passage, Quantity


@dataclass(frozen=True)
class AttributeRule:
    name: str
    label: str
    units: frozenset[str]
    context: re.Pattern[str] | None = None


ATTRIBUTE_RULES: tuple[AttributeRule, ...] = (
    AttributeRule("vram", "mémoire graphique", frozenset({"GB"}),
                  re.compile(r"\b(?:vram|gddr\d*|m[ée]moire graphique|graphics memory)\b", re.IGNORECASE)),
    AttributeRule("ram", "mémoire vive", frozenset({"GB"}),
                  re.compile(r"\b(?:ram|m[ée]moire vive|ddr\d*|lpddr\d*|memory)\b", re.IGNORECASE)),
    AttributeRule("storage", "stockage", frozenset({"GB"}),
                  re.compile(r"\b(?:ssd|hdd|nvme|stockage|storage|disque|emmc)\b", re.IGNORECASE)),
    AttributeRule("price", "prix", frozenset({"EUR", "USD"})),
    AttributeRule("frequency", "fréquence", frozenset({"GHz"})),
    AttributeRule("screen", "écran", frozenset({"inch"})),
    AttributeRule("power", "puissance", frozenset({"W"})),
    AttributeRule("cores", "cœurs", frozenset({"cores"})),
    AttributeRule("weight", "poids", frozenset({"kg"})),
)
ATTRIBUTE_LABELS: dict[str, str] = {rule.name: rule.label for rule in ATTRIBUTE_RULES}
ATTRIBUTE_LABELS.update({"value": "valeur", "type": "type", "capacity": "capacité", "statement": "énoncé"})

_FACT_VERBS = (
    r"co[uû]te|co[uû]tent|vaut|valent|est|sont|[ée]tait|dispose|disposent|poss[èe]de|poss[èe]dent|a|ont|"
    r"embarque|embarquent|offre|offrent|propose|proposent|int[èe]gre|int[èe]grent|affiche|p[èe]se|mesure|"
    r"consomme|atteint|se vend|is|are|was|were|costs?|has|have|offers?|features?|weighs|includes?|comes|"
    r"sells|retails"
)
_REQUIREMENT_VERBS = (
    r"n[ée]cessite|n[ée]cessitent|requiert|requi[èe]rent|exige|exigent|demande|demandent|recommande|"
    r"recommandent|requires?|needs?|recommends?"
)
_VERB_RE = re.compile(rf"\b(?:{_REQUIREMENT_VERBS}|{_FACT_VERBS})\b", re.IGNORECASE)
_REQUIREMENT_RE = re.compile(
    rf"\b(?:{_REQUIREMENT_VERBS}|configuration minimale|minimum requis|minimum requirements?)\b", re.IGNORECASE
)
_GE_RE = re.compile(
    r"(?:au moins|minimum|au minimum|plus de|at least|more than|≥|>=|à partir de|a partir de|from)\s*(?:de\s*|of\s*)?$",
    re.IGNORECASE,
)
_LE_RE = re.compile(
    r"(?:au plus|maximum|au maximum|moins de|jusqu'à|jusqu’à|jusqu'a|at most|less than|up to|under|"
    r"≤|<=|inf[ée]rieur à|inf[ée]rieur a|pas plus de|below)\s*(?:de\s*)?$",
    re.IGNORECASE,
)
_TYPE_RE = re.compile(
    r"^(?P<subject>.+?)\s+(?:est|is)\s+(?:un|une|a|an)\s+(?P<type>[^,.;:!?]+)", re.IGNORECASE
)
_TYPE_CUT_RE = re.compile(r"\s+(?:de\s+\d|avec|qui|que|with|that|which|pour|for|à\s+\d)\b.*$", re.IGNORECASE)
_LABEL_VALUE_RE = re.compile(r"^(?P<label>[^:]{2,60}?)\s*:\s*", re.IGNORECASE)
_WINDOW = 32


def _claim_id(passage_id: str, sentence_index: int, attribute: str, value: str) -> str:
    raw = f"{passage_id}|{sentence_index}|{attribute}|{value}".encode()
    return "c_" + hashlib.blake2b(raw, digest_size=6).hexdigest()


def attribute_rule_for(quantity: Quantity, sentence: str, start: int, end: int) -> AttributeRule | None:
    """Règle d'attribut dont le mot-clé est le plus proche de la quantité (après, puis avant)."""
    after = sentence[end : end + _WINDOW]
    before_start = max(0, start - _WINDOW)
    before = sentence[before_start:start]
    best: tuple[float, AttributeRule] | None = None
    for rule in ATTRIBUTE_RULES:
        if quantity.unit not in rule.units:
            continue
        if rule.context is None:
            return rule
        hit_after = rule.context.search(after)
        if hit_after is not None and (best is None or hit_after.start() < best[0]):
            best = (hit_after.start(), rule)
        for hit_before in rule.context.finditer(before):
            # À distance égale, le mot qui suit la quantité l'emporte (« 512 Go SSD », « 16 Go de RAM »).
            distance = len(before) - hit_before.end() + 0.5
            if best is None or distance < best[0]:
                best = (distance, rule)
    return best[1] if best else None


def _comparator_for(sentence: str, start: int, requirement: bool) -> Comparator:
    before = sentence[max(0, start - _WINDOW) : start].rstrip()
    if _GE_RE.search(before):
        return "ge"
    if _LE_RE.search(before):
        return "le"
    return "ge" if requirement else "eq"


def _subject_of(sentence: str) -> tuple[str | None, int]:
    """(sujet nettoyé ou None, fin de la zone sujet) : texte avant le premier verbe ou avant « : »."""
    match = _VERB_RE.search(sentence)
    if match is not None:
        return clean_subject(sentence[: match.start()]), match.start()
    label = _LABEL_VALUE_RE.match(sentence)
    if label is not None:
        return clean_subject(label.group("label")), label.end()
    return None, 0


@dataclass
class RuleExtractor:
    """Extracteur déterministe ; `focus` suit l'entité courante d'un document à l'autre passage."""

    def extract(self, passage: Passage, focus: str | None = None) -> tuple[list[Claim], str | None]:
        claims: list[Claim] = []
        for index, sentence in enumerate(split_sentences(passage.text)):
            sentence_claims, focus = self._sentence(passage, index, sentence, focus)
            claims.extend(sentence_claims)
        return claims, focus

    def _sentence(
        self, passage: Passage, index: int, sentence: str, focus: str | None
    ) -> tuple[list[Claim], str | None]:
        claims: list[Claim] = []
        type_match = _TYPE_RE.match(sentence)
        if type_match:
            subject = clean_subject(type_match.group("subject")) or focus
            type_value = _TYPE_CUT_RE.sub("", type_match.group("type")).strip()
            if subject and type_value and len(type_value.split()) <= 6:
                focus = subject
                claims.append(self._make(passage, index, sentence, subject, "type", type_value, None, "eq", "type"))

        quantities = [(m, q) for m, q in iter_quantities(sentence) if q.unit]
        if not quantities:
            return claims, focus
        requirement = bool(_REQUIREMENT_RE.search(sentence))
        explicit, _ = _subject_of(sentence)
        subject = explicit or focus
        if explicit is not None:
            focus = explicit
        if subject is None:
            return claims, focus
        subject_start = sentence.find(explicit) if explicit else -1

        for match, quantity in quantities:
            if explicit and subject_start <= match.start() < subject_start + len(explicit):
                continue  # numéro de modèle dans le nom (« 15 pouces » d'un « Pavilion 15 pouces »)
            rule = attribute_rule_for(quantity, sentence, match.start(), match.end())
            attribute = rule.name if rule else ("capacity" if quantity.unit == "GB" else "value")
            comparator = _comparator_for(sentence, match.start(), requirement)
            kind: ClaimKind = "requirement" if requirement else "fact"
            claims.append(
                self._make(passage, index, sentence, subject, attribute, match.group(0).strip(), quantity,
                           comparator, kind)
            )
        return claims, focus

    @staticmethod
    def _make(
        passage: Passage,
        index: int,
        sentence: str,
        subject: str,
        attribute: str,
        value_text: str,
        quantity: Quantity | None,
        comparator: Comparator,
        kind: ClaimKind,
    ) -> Claim:
        return Claim(
            id=_claim_id(passage.id, index, attribute, value_text),
            subject=subject,
            subject_key=entity_key(subject),
            attribute=attribute,
            value_text=value_text,
            quantity=quantity,
            comparator=comparator,
            kind=kind,
            quote=sentence,
            passage_id=passage.id,
            source=passage.source,
            extractor="rules",
        )
