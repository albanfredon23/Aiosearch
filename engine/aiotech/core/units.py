"""
Nombres et unités : lecture FR/EN, unités canoniques, comparaison avec tolérance.

Les contraintes (« budget de 500 € », « au moins 16 Go de RAM ») et les affirmations
extraites des sources sont ramenées à la même unité canonique avant toute comparaison :
« 2 semaines » et « 14 jours » sont la même valeur, « 1 To » vaut 1000 Go.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from aiotech.models import Comparator, Quantity

NUMBER_PATTERN = (
    r"\d{1,3}(?:,\d{3})+(?:\.\d+)?"
    r"|\d{1,3}(?:[ \u00a0\u202f]\d{3})+(?:[.,]\d+)?"
    r"|\d+(?:[.,]\d+)?"
)


@dataclass(frozen=True)
class UnitSpec:
    canonical: str
    factor: float
    dimension: str


_UNIT_ALIASES: dict[str, UnitSpec] = {}


def _register(aliases: tuple[str, ...], canonical: str, factor: float, dimension: str) -> None:
    for alias in aliases:
        _UNIT_ALIASES[alias] = UnitSpec(canonical, factor, dimension)


_register(("€", "eur", "euro", "euros"), "EUR", 1.0, "currency_eur")
_register(("$", "usd", "dollar", "dollars"), "USD", 1.0, "currency_usd")
_register(("go", "gb", "gio", "gigaoctet", "gigaoctets", "giga-octets", "giga"), "GB", 1.0, "data")
_register(("to", "tb", "tio", "teraoctet", "teraoctets", "téraoctet", "téraoctets"), "GB", 1000.0, "data")
_register(("mo", "mb", "megaoctet", "megaoctets", "mégaoctets"), "GB", 0.001, "data")
_register(("ghz",), "GHz", 1.0, "frequency")
_register(("mhz",), "GHz", 0.001, "frequency")
_register(("jour", "jours", "day", "days", "j"), "day", 1.0, "duration")
_register(("semaine", "semaines", "week", "weeks"), "day", 7.0, "duration")
_register(("mois", "month", "months"), "day", 30.44, "duration")
_register(("an", "ans", "année", "années", "annee", "annees", "year", "years"), "day", 365.25, "duration")
_register(("heure", "heures", "hour", "hours", "h"), "hour", 1.0, "time")
_register(("minute", "minutes", "min"), "hour", 1.0 / 60.0, "time")
_register(("%", "pour cent", "pourcent", "percent"), "%", 1.0, "ratio")
_register(("kg", "kilo", "kilos", "kilogramme", "kilogrammes"), "kg", 1.0, "mass")
_register(("g", "gramme", "grammes", "grams", "gram"), "kg", 0.001, "mass")
_register(("mg", "milligramme", "milligrammes"), "kg", 1e-6, "mass")
_register(("tonne", "tonnes"), "kg", 1000.0, "mass")
_register(("km", "kilomètre", "kilomètres", "kilometre", "kilometres", "kilometer", "kilometers"), "m", 1000.0, "length")
_register(("mètre", "mètres", "metre", "metres", "meter", "meters"), "m", 1.0, "length")
_register(("cm", "centimètre", "centimètres"), "m", 0.01, "length")
_register(("mm", "millimètre", "millimètres"), "m", 0.001, "length")
_register(("w", "watt", "watts"), "W", 1.0, "power")
_register(("kw", "kilowatt", "kilowatts"), "W", 1000.0, "power")
_register(("cœurs", "coeurs", "cores", "core", "cœur", "coeur"), "cores", 1.0, "count")
_register(("pouces", "pouce", "inch", "inches", '"'), "inch", 1.0, "length_inch")

# Alternatives triées du plus long au plus court : « gigaoctets » avant « go », « kg » avant « g ».
_UNIT_ALTERNATION = "|".join(
    re.escape(alias) for alias in sorted(_UNIT_ALIASES, key=len, reverse=True)
)
QUANTITY_RE = re.compile(
    rf"(?:(?P<prefix_currency>[$€])\s?)?(?P<number>{NUMBER_PATTERN})\s?(?P<unit>(?:{_UNIT_ALTERNATION})(?![a-zà-ÿ]))?",
    re.IGNORECASE,
)

_DISPLAY: dict[str, str] = {
    "EUR": "€", "USD": "$", "GB": "Go", "GHz": "GHz", "day": "jours", "hour": "h", "%": "%",
    "kg": "kg", "m": "m", "W": "W", "cores": "cœurs", "inch": "pouces",
}


def parse_number(raw: str) -> float:
    """Lit « 1 299,99 », « 1,299.99 », « 4,5 », « 16 » ; lève ValueError sinon."""
    text = re.sub(r"[ \u00a0\u202f]", "", raw.strip())
    if "," in text and "." in text:
        decimal = "," if text.rfind(",") > text.rfind(".") else "."
        thousands = "." if decimal == "," else ","
        text = text.replace(thousands, "").replace(decimal, ".")
    elif "," in text:
        head, _, tail = text.rpartition(",")
        text = head.replace(",", "") + ("" if len(tail) == 3 and head.isdigit() and head != "0" and len(head) <= 3 else ".") + tail
    return float(text)


def unit_spec(unit_text: str) -> UnitSpec | None:
    return _UNIT_ALIASES.get(unit_text.strip().lower())


def make_quantity(number_text: str, unit_text: str | None) -> Quantity:
    value = parse_number(number_text)
    if not unit_text:
        return Quantity(value=value, unit="")
    spec = unit_spec(unit_text)
    if spec is None:
        return Quantity(value=value, unit=unit_text.strip().lower())
    return Quantity(value=round(value * spec.factor, 6), unit=spec.canonical)


# Alias ambigus en minuscules : « from 2 to 5 » ne doit pas devenir 2 To.
_UPPERCASE_ONLY = frozenset({"to", "tb", "mo", "mb"})


def iter_quantities(text: str) -> list[tuple[re.Match[str], Quantity]]:
    """Toutes les quantités d'un texte, avec leur position, unités canoniques appliquées."""
    found: list[tuple[re.Match[str], Quantity]] = []
    for match in QUANTITY_RE.finditer(text):
        unit = match.group("unit") or match.group("prefix_currency")
        if unit and unit.lower() in _UPPERCASE_ONLY and not unit[0].isupper():
            unit = None
        try:
            found.append((match, make_quantity(match.group("number"), unit)))
        except ValueError:
            continue
    return found


def comparable(a: Quantity, b: Quantity) -> bool:
    return a.unit == b.unit


def same_value(a: Quantity, b: Quantity, rel_tol: float = 0.02) -> bool:
    """Égalité à tolérance relative près (arrondis, conversions de durées)."""
    if not comparable(a, b):
        return False
    scale = max(abs(a.value), abs(b.value), 1e-9)
    return abs(a.value - b.value) <= rel_tol * scale


def satisfies(value: Quantity, comparator: Comparator, bound: Quantity) -> bool | None:
    """Vrai / faux si les unités sont comparables, None sinon (inconnu)."""
    if not comparable(value, bound):
        return None
    if comparator == "eq":
        return same_value(value, bound)
    if comparator == "le":
        return value.value <= bound.value + 1e-9
    if comparator == "lt":
        return value.value < bound.value - 1e-9
    if comparator == "ge":
        return value.value >= bound.value - 1e-9
    return value.value > bound.value + 1e-9


def format_number(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        integer = round(value)
        return f"{integer:,}".replace(",", " ")
    return f"{value:,.2f}".replace(",", " ").replace(".", ",").rstrip("0").rstrip(",")


def format_quantity(q: Quantity) -> str:
    display = _DISPLAY.get(q.unit, q.unit)
    number = format_number(q.value)
    if q.unit == "USD":
        return f"{number} $"
    return f"{number} {display}".strip()


COMPARATOR_LABELS: dict[str, str] = {"eq": "=", "le": "≤", "lt": "<", "ge": "≥", "gt": ">"}
