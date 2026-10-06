from __future__ import annotations

import os
from pathlib import Path

import pytest

from aiotech.gateway.tools import FileJail, ToolError, ToolRegistry, ToolSpec, calculate, default_registry


@pytest.mark.parametrize(
    ("expression", "expected"),
    [("2+2*3", 8), ("(1+2)**10", 59049), ("4,5*2", 9), ("max(1,2,3)", 3), ("sqrt(16)", 4), ("10/4", 2.5)],
)
def test_calculator_evaluates_arithmetic(expression: str, expected: float) -> None:
    assert calculate(expression)["result"] == expected


@pytest.mark.parametrize(
    "expression",
    ["__import__('os').system('id')", "().__class__", "open('/etc/passwd')", "2**1000", "1/0", "x" * 300,
     "[1,2,3]", "lambda: 1", "eval('1')"],
)
def test_calculator_refuses_code(expression: str) -> None:
    with pytest.raises(ToolError):
        calculate(expression)


@pytest.fixture
def jail(tmp_path: Path) -> FileJail:
    root = tmp_path / "docs"
    root.mkdir()
    (root / "notes.txt").write_text("bonjour", encoding="utf-8")
    (tmp_path / ".env").write_text("SECRET=1", encoding="utf-8")
    (root / ".env").write_text("SECRET=2", encoding="utf-8")
    (root / "cle.pem").write_text("-----BEGIN-----", encoding="utf-8")
    (root / "gros.txt").write_text("a" * 300_000, encoding="utf-8")
    os.symlink(tmp_path / ".env", root / "lien.txt")
    return FileJail(root)


def test_jail_reads_allowed_file(jail: FileJail) -> None:
    assert jail.read("notes.txt")["content"] == "bonjour"


@pytest.mark.parametrize(
    "path", ["../.env", ".env", "/etc/passwd", "lien.txt", "cle.pem", "sous/../notes.txt", "gros.txt", "absent.txt",
             "notes.txt\x00"],
)
def test_jail_refuses_escapes_and_secrets(jail: FileJail, path: str) -> None:
    with pytest.raises(ToolError):
        jail.read(path)


def test_registry_forbids_code_execution_tools() -> None:
    registry = ToolRegistry()

    async def handler(args: dict[str, object]) -> dict[str, object]:
        return {}

    with pytest.raises(ValueError):
        registry.register(ToolSpec(name="run_python", description="", input_schema={}, handler=handler))
    names = set(default_registry(None).tools)
    assert names == {"calculate"}
    assert not names & {"run_python", "exec", "shell", "bash"}


async def test_registry_schemas_and_calls() -> None:
    registry = default_registry(None)
    assert registry.anthropic_schemas()[0]["name"] == "calculate"
    assert registry.openai_schemas()[0]["function"]["name"] == "calculate"
    assert (await registry.call("calculate", {"expression": "6*7"}))["result"] == 42
    with pytest.raises(ToolError):
        await registry.call("run_python", {})
