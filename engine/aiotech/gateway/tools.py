"""
Outils exposés aux LLM et aux clients MCP : environnements confinés uniquement.

Aucun outil d'exécution de code arbitraire (pas de run_python, pas de shell) :
    calculate   calculatrice statique : l'expression est analysée en arbre syntaxique et
                seuls les nombres, + − × ÷ // % ** (exposant borné), parenthèses et une
                liste fermée de fonctions mathématiques sont évalués ;
    read_file   lecture emprisonnée dans un répertoire racine (chroot virtuel) : chemin
                résolu puis comparé à la racine, liens symboliques suivis puis revérifiés,
                fichiers cachés (.env, .git...) et extensions sensibles refusés, taille bornée ;
    search      la recherche AIOTECH elle-même (branchée par l'API).
"""
from __future__ import annotations

import ast
import math
import operator
import os
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MAX_EXPRESSION_CHARS = 200
MAX_EXPONENT = 64
MAX_MAGNITUDE = 1e100
MAX_FILE_BYTES = 256 * 1024
_FORBIDDEN_SUFFIXES = frozenset({".env", ".pem", ".key", ".p12", ".pfx", ".sqlite3", ".db", ".kdbx"})
_FORBIDDEN_NAMES = frozenset({"id_rsa", "id_ed25519", "credentials", "secrets", "passwd", "shadow"})


class ToolError(ValueError):
    """Refus explicite d'un outil, renvoyé tel quel au client."""


_BINARY: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY: dict[type[ast.unaryop], Callable[[float], float]] = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCTIONS: dict[str, Callable[..., float]] = {
    "sqrt": math.sqrt, "abs": abs, "round": round, "min": min, "max": max, "log": math.log, "log10": math.log10,
    "exp": math.exp, "sin": math.sin, "cos": math.cos, "tan": math.tan, "floor": math.floor, "ceil": math.ceil,
}
_CONSTANTS: dict[str, float] = {"pi": math.pi, "e": math.e}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float) and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.Name) and node.id in _CONSTANTS:
        return _CONSTANTS[node.id]
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise ToolError(f"exposant limité à {MAX_EXPONENT}")
        result = _BINARY[type(node.op)](left, right)
        if isinstance(result, complex) or abs(result) > MAX_MAGNITUDE:
            raise ToolError("résultat hors limites")
        return float(result)
    if (
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCTIONS
        and not node.keywords and len(node.args) <= 4
    ):
        return float(_FUNCTIONS[node.func.id](*[_eval(a) for a in node.args]))
    raise ToolError("expression refusée : seuls les nombres, opérateurs et fonctions mathématiques sont permis")


def calculate(expression: str) -> dict[str, Any]:
    text = expression.strip().replace("×", "*").replace("÷", "/").replace("^", "**")
    if not any(ch.isalpha() for ch in text):
        text = re.sub(r"(?<=\d),(?=\d)", ".", text)
    if not text or len(text) > MAX_EXPRESSION_CHARS:
        raise ToolError(f"expression vide ou trop longue (max {MAX_EXPRESSION_CHARS} caractères)")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ToolError("expression mal formée") from exc
    try:
        value = _eval(tree)
    except (ZeroDivisionError, OverflowError, ValueError) as exc:
        if isinstance(exc, ToolError):
            raise
        raise ToolError(f"calcul impossible : {exc}") from exc
    rounded = round(value, 12)
    return {"expression": expression, "result": int(rounded) if rounded.is_integer() else rounded}


@dataclass(frozen=True)
class FileJail:
    root: Path
    max_bytes: int = MAX_FILE_BYTES

    def resolve(self, relative: str) -> Path:
        if not relative or "\x00" in relative or len(relative) > 512:
            raise ToolError("chemin invalide")
        root = self.root.resolve(strict=True)
        candidate = Path(relative)
        if candidate.is_absolute() or candidate.drive:
            raise ToolError("chemin absolu refusé : donner un chemin relatif au dossier autorisé")
        parts = candidate.parts
        if any(p == ".." for p in parts):
            raise ToolError("remontée de dossier (..) refusée")
        if any(p.startswith(".") for p in parts):
            raise ToolError("fichiers et dossiers cachés refusés (.env, .git...)")
        target = (root / candidate).resolve(strict=False)
        if target != root and root not in target.parents:
            raise ToolError("accès hors du dossier autorisé refusé")
        name = target.name.lower()
        if target.suffix.lower() in _FORBIDDEN_SUFFIXES or name in _FORBIDDEN_NAMES or name.startswith(".env"):
            raise ToolError("type de fichier sensible refusé")
        return target

    def read(self, relative: str) -> dict[str, Any]:
        target = self.resolve(relative)
        if not target.is_file():
            raise ToolError("fichier introuvable")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(target, flags)
        try:
            stat = os.fstat(fd)
            if stat.st_size > self.max_bytes:
                raise ToolError(f"fichier trop volumineux (max {self.max_bytes} octets)")
            data = os.read(fd, self.max_bytes + 1)
        finally:
            os.close(fd)
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ToolError("fichier binaire refusé") from exc
        return {"path": relative, "bytes": len(data), "content": text}


ToolHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: ToolHandler


@dataclass
class ToolRegistry:
    tools: dict[str, ToolSpec] = field(default_factory=dict)

    def register(self, spec: ToolSpec) -> None:
        if spec.name in {"run_python", "exec", "shell", "bash", "eval"}:
            raise ValueError("les outils d'exécution de code arbitraire sont interdits")
        self.tools[spec.name] = spec

    async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        spec = self.tools.get(name)
        if spec is None:
            raise ToolError(f"outil inconnu : {name}")
        return await spec.handler(arguments)

    def anthropic_schemas(self) -> list[dict[str, Any]]:
        return [{"name": t.name, "description": t.description, "input_schema": t.input_schema}
                for t in self.tools.values()]

    def openai_schemas(self) -> list[dict[str, Any]]:
        return [{"type": "function", "function": {"name": t.name, "description": t.description,
                                                  "parameters": t.input_schema}}
                for t in self.tools.values()]


def default_registry(jail: FileJail | None) -> ToolRegistry:
    registry = ToolRegistry()

    async def calc(args: dict[str, Any]) -> dict[str, Any]:
        return calculate(str(args.get("expression", "")))

    registry.register(ToolSpec(
        name="calculate",
        description="Calculatrice confinée : opérations arithmétiques et fonctions mathématiques usuelles.",
        input_schema={"type": "object", "properties": {"expression": {"type": "string", "maxLength": 200}},
                      "required": ["expression"], "additionalProperties": False},
        handler=calc,
    ))
    if jail is not None:
        active_jail = jail

        async def read_file(args: dict[str, Any]) -> dict[str, Any]:
            return active_jail.read(str(args.get("path", "")))

        registry.register(ToolSpec(
            name="read_file",
            description="Lit un fichier texte du dossier documentaire autorisé (chemin relatif, lecture seule).",
            input_schema={"type": "object", "properties": {"path": {"type": "string", "maxLength": 512}},
                          "required": ["path"], "additionalProperties": False},
            handler=read_file,
        ))
    return registry
