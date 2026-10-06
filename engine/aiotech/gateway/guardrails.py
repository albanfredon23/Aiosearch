"""
Garde-fous d'entrée : masquage des données personnelles et filtrage des injections.

Ordre imposé, AVANT toute planification, recherche ou appel LLM :
    1. limites de taille et caractères de contrôle ;
    2. détection d'injection de prompt (français et anglais) sur une forme normalisée
       du texte (Unicode NFKC, caractères invisibles retirés, accents retirés,
       substitutions « l33t » courantes) : une requête bloquée s'arrête ici et ne coûte
       aucun token ;
    3. masquage RGPD des données sensibles :
         cartes bancaires   13 à 19 chiffres ET contrôle de Luhn valide ;
         IBAN               format pays + clé ET contrôle modulo 97 (ISO 13616) ;
         adresses IP        IPv4 et IPv6 validées par le module `ipaddress` ;
         téléphones         formats français (0X, +33, 0033) et internationaux E.164 ;
         e-mails.
Les passages récupérés (web, documents) passent par le même détecteur : un passage
qui contient une injection indirecte est écarté du contexte envoyé au LLM.
"""
from __future__ import annotations

import ipaddress
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

from aiotech.models import PiiFinding

MAX_QUERY_CHARS = 2000


class PiiType(StrEnum):
    CARD = "CARTE_BANCAIRE"
    IBAN = "IBAN"
    EMAIL = "EMAIL"
    IP = "ADRESSE_IP"
    PHONE = "TELEPHONE"


def luhn_valid(digits: str) -> bool:
    if not digits.isdigit() or not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def iban_valid(raw: str) -> bool:
    iban = re.sub(r"\s+", "", raw).upper()
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{11,30}", iban):
        return False
    rearranged = iban[4:] + iban[:4]
    numeric = "".join(str(int(ch, 36)) for ch in rearranged)
    return int(numeric) % 97 == 1


_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,24}\b")
_IBAN_RE = re.compile(r"\b[A-Za-z]{2}\d{2}(?:[ ]?[A-Za-z0-9]){11,30}\b")
_CARD_RE = re.compile(r"(?<![\d.,])\d(?:[ \-]?\d){12,18}(?![\d.,]\d)")
_IPV4_RE = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
_IPV6_RE = re.compile(r"(?<![0-9A-Fa-f:])(?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{0,4}(?![0-9A-Fa-f:])")
_PHONE_RE = re.compile(
    r"(?<![\w+])(?:(?:\+|00)33[ .\-]?[1-9](?:[ .\-]?\d{2}){4}|0[1-9](?:[ .\-]?\d{2}){4}|"
    r"\+(?:[1-9]\d{0,2})[ .\-]?\d{2,4}(?:[ .\-]?\d{2,4}){2,4})(?!\w)"
)


Acceptor = Callable[[str], bool] | None


def _replace(pattern: re.Pattern[str], text: str, label: PiiType, accept: Acceptor) -> tuple[str, int]:
    count = 0

    def sub(match: re.Match[str]) -> str:
        nonlocal count
        value = match.group(0)
        if accept is not None and not accept(value):
            return value
        count += 1
        return f"[{label.value}]"

    return pattern.sub(sub, text), count


def _is_ip(value: str) -> bool:
    try:
        ipaddress.ip_address(value)
    except ValueError:
        return False
    return True


def _is_ipv6(value: str) -> bool:
    return ":" in value and value.count(":") >= 2 and _is_ip(value) and any(c.isalnum() for c in value)


def mask_pii(text: str) -> tuple[str, list[PiiFinding]]:
    """Texte masqué et comptage par type ; les valeurs masquées ne sont jamais renvoyées."""
    findings: list[PiiFinding] = []
    steps: list[tuple[re.Pattern[str], PiiType, Acceptor]] = [
        (_EMAIL_RE, PiiType.EMAIL, None),
        (_IBAN_RE, PiiType.IBAN, iban_valid),
        (_CARD_RE, PiiType.CARD, lambda v: luhn_valid(re.sub(r"[ \-]", "", v))),
        (_IPV6_RE, PiiType.IP, _is_ipv6),
        (_IPV4_RE, PiiType.IP, _is_ip),
        (_PHONE_RE, PiiType.PHONE, None),
    ]
    counts: dict[PiiType, int] = {}
    for pattern, label, accept in steps:
        text, n = _replace(pattern, text, label, accept)
        if n:
            counts[label] = counts.get(label, 0) + n
    for label, n in counts.items():
        findings.append(PiiFinding(type=label.value, count=n))
    return text, findings


_INVISIBLE = re.compile(r"[​-‏‪-‮⁠-⁤﻿­]")
_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})


def normalize_for_screening(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _INVISIBLE.sub("", text)
    text = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    text = text.lower().translate(_LEET)
    text = re.sub(r"[_*~`|]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass(frozen=True)
class InjectionRule:
    name: str
    pattern: re.Pattern[str]


def _rule(name: str, pattern: str) -> InjectionRule:
    return InjectionRule(name, re.compile(pattern, re.IGNORECASE))


INJECTION_RULES: tuple[InjectionRule, ...] = (
    _rule("override_en", r"\b(?:ignore|disregard|forget|override|skip)\b.{0,30}\b(?:previous|prior|above|earlier|all|any|"
                         r"your|the|these|those)\b.{0,20}\b(?:instructions?|rules?|prompts?|directives?|guidelines?|"
                         r"constraints?)\b"),
    _rule("override_fr", r"\b(?:ignore[rsz]?|oublie[rsz]?|neglige[rsz]?|outrepasse[rsz]?|ne tiens? pas compte)\b.{0,30}"
                         r"\b(?:instructions?|consignes?|regles?|directives?|prompts?|restrictions?)\b"),
    _rule("persona", r"\b(?:you are now|from now on,? you|act as (?:an? )?(?:unrestricted|unfiltered|evil|jailbroken)|"
                     r"pretend (?:to be|you are)|tu es (?:desormais|maintenant)|a partir de maintenant,? tu|"
                     r"fais comme si tu etais|agis (?:comme|en tant que) (?:un|une)? ?(?:ia|assistant|modele)? ?"
                     r"(?:sans|libre|non filtre))"),
    _rule("prompt_leak", r"\b(?:(?:reveal|show|print|repeat|display|leak|output|give|tell)(?: me| us)? (?:your|the) "
                         r"(?:full |exact |original )?(?:system |initial |hidden |secret )?(?:prompt|instructions?)|"
                         r"what (?:is|are) your (?:system |initial |hidden )?(?:prompt|instructions?)|"
                         r"(?:revele|affiche|repete|montre|donne|ecris|recopie)[sz]?(?:-moi| moi)? (?:ton|tes|votre|vos|le|les) "
                         r"(?:prompt|instructions?|consignes?)(?: (?:systeme|initiales?|cachees?|secretes?))?)\b"),
    _rule("jailbreak", r"\b(?:jailbreak|developer mode|mode developpeur|dan mode|do anything now|sans (?:aucune )?"
                       r"(?:restriction|limite|filtre|censure)s?|without (?:any )?(?:restrictions?|limits?|filters?|"
                       r"censorship)|no (?:restrictions|filters|rules)|(?:bypass|contourne[rz]?) (?:les |the |your |tes )?"
                       r"(?:filtres?|filters?|regles?|rules?|securites?|safety|garde-fous?|guardrails?))\b"),
    _rule("role_markup", r"(?:<\|im_start\|>|<\|im_end\|>|<\|system\|>|\[/?inst\]|<</?sys>>|^#{2,} ?system\b|"
                         r"(?:^|\n|\s)(?:system|systeme|assistant)\s*:\s*(?:you|tu|ignore|oublie))"),
    _rule("secret_exfiltration", r"\b(?:give|send|print|show|reveal|leak|dump|donne|envoie|affiche|revele|montre)[sz]?\b"
                                 r".{0,40}\b(?:api[ -]?keys?|cles? (?:d'?)?api|secrets?|passwords?|mots? de passe|"
                                 r"tokens?|\.env|credentials?|identifiants?)\b"),
    _rule("code_execution", r"(?:\brun_python\b|\bexec\s*\(|\beval\s*\(|\bos\.system\b|\bsubprocess\b|\b__import__\b|"
                            r"\brm -rf\b|/etc/(?:passwd|shadow)\b|(?:\.\./){2,})"),
)


@dataclass(frozen=True)
class InjectionVerdict:
    blocked: bool
    rules: tuple[str, ...] = ()


def detect_injection(text: str) -> InjectionVerdict:
    raw = _INVISIBLE.sub("", unicodedata.normalize("NFKC", text)).lower()
    variants = {
        raw,
        normalize_for_screening(text),
        normalize_for_screening(re.sub(r"(?<=\w)[.\-](?=\w)", "", text)),
    }
    hits = tuple(dict.fromkeys(r.name for r in INJECTION_RULES for v in variants if r.pattern.search(v)))
    return InjectionVerdict(blocked=bool(hits), rules=hits)


@dataclass(frozen=True)
class Screening:
    text: str
    blocked: bool
    reason: str | None = None
    rules: tuple[str, ...] = ()
    pii: tuple[PiiFinding, ...] = field(default_factory=tuple)


class Guardrails:
    def __init__(self, max_chars: int = MAX_QUERY_CHARS) -> None:
        self.max_chars = max_chars

    def screen(self, query: str) -> Screening:
        cleaned = "".join(ch for ch in query if ch in "\n\t" or unicodedata.category(ch)[0] != "C" or ch == "‍")
        cleaned = cleaned.strip()
        if not cleaned:
            return Screening(text="", blocked=True, reason="Requête vide")
        if len(cleaned) > self.max_chars:
            return Screening(text="", blocked=True, reason=f"Requête trop longue (plus de {self.max_chars} caractères)")
        verdict = detect_injection(query)
        if verdict.blocked:
            return Screening(text="", blocked=True, reason="Tentative d'injection de prompt bloquée", rules=verdict.rules)
        masked, findings = mask_pii(cleaned)
        return Screening(text=masked, blocked=False, pii=tuple(findings))

    @staticmethod
    def passage_is_safe(text: str) -> bool:
        return not detect_injection(text).blocked

    @staticmethod
    def mask(text: str) -> str:
        return mask_pii(text)[0]
