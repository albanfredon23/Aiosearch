"""
Contraintes explicites d'une requête : « PC à 500 € », « au moins 16 Go de RAM »,
« moins de 2 kg », « under $800 ».

Règles de comparaison par défaut, quand la requête ne précise rien :
    montant (€, $)                     -> borne haute (un budget) : ≤
    mémoire, stockage, cœurs, GHz      -> borne basse (un minimum requis) : ≥
    autres unités                      -> égalité à tolérance près
Les qualificatifs explicites (« au moins », « maximum », « moins de »...) l'emportent.
"""
from __future__ import annotations

import re

from aiotech.core.units import COMPARATOR_LABELS, format_quantity, iter_quantities
from aiotech.graph.claims import ATTRIBUTE_LABELS, attribute_rule_for
from aiotech.models import Comparator, Constraint, Quantity

_GE_BEFORE = re.compile(
    r"(?:au moins|minimum|au minimum|plus de|at least|more than|≥|>=|à partir de|a partir de|over|above)\s*(?:de\s*|of\s*)?$",
    re.IGNORECASE,
)
_LE_BEFORE = re.compile(
    r"(?:au plus|maximum|au maximum|moins de|jusqu'à|jusqu’à|jusqu'a|at most|less than|up to|under|"
    r"≤|<=|inf[ée]rieur à|inf[ée]rieur a|pas plus de|below)\s*(?:de\s*)?$",
    re.IGNORECASE,
)
_GE_AFTER = re.compile(r"^\s*(?:minimum|min|ou plus|au moins|or more|at least)\b", re.IGNORECASE)
_LE_AFTER = re.compile(r"^\s*(?:maximum|max|au plus|ou moins|or less|at most|tout compris)\b", re.IGNORECASE)
_LOWER_BOUND_UNITS = frozenset({"GB", "cores", "GHz"})
_WINDOW = 28


def _attribute(quantity: Quantity, text: str, start: int, end: int) -> str:
    rule = attribute_rule_for(quantity, text, start, end)
    return rule.name if rule else f"unit:{quantity.unit}"


def _comparator(quantity: Quantity, text: str, start: int, end: int) -> Comparator:
    before = text[max(0, start - _WINDOW) : start].rstrip()
    after = text[end : end + 16]
    if _GE_BEFORE.search(before) or _GE_AFTER.search(after):
        return "ge"
    if quantity.unit in {"EUR", "USD"}:
        return "le"
    if _LE_BEFORE.search(before) or _LE_AFTER.search(after):
        return "le"
    if quantity.unit in _LOWER_BOUND_UNITS:
        return "ge"
    return "eq"


def attribute_label(attribute: str) -> str:
    if attribute.startswith("unit:"):
        return "valeur"
    return ATTRIBUTE_LABELS.get(attribute, attribute)


def constraint_label(attribute: str, comparator: Comparator, quantity: Quantity) -> str:
    return f"{attribute_label(attribute)} {COMPARATOR_LABELS[comparator]} {format_quantity(quantity)}"


def extract_constraints(query: str) -> list[Constraint]:
    constraints: list[Constraint] = []
    seen: set[tuple[str, str, float]] = set()
    for match, quantity in iter_quantities(query):
        if not quantity.unit:
            continue
        start, end = match.start(), match.end()
        attribute = _attribute(quantity, query, start, end)
        comparator = _comparator(quantity, query, start, end)
        key = (attribute, comparator, quantity.value)
        if key in seen:
            continue
        seen.add(key)
        constraints.append(
            Constraint(
                attribute=attribute,
                comparator=comparator,
                quantity=quantity,
                label=constraint_label(attribute, comparator, quantity),
                origin="query",
            )
        )
    return constraints
