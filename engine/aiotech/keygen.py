"""
Secrets générés au premier démarrage (docker compose) : clé d'API de l'interface web et clé
de session SearXNG. Un fichier déjà présent n'est jamais écrasé ; aucun secret n'est affiché.

    python -m aiotech.keygen /run/aiotech
"""
from __future__ import annotations

import os
import secrets
import sys
from pathlib import Path

FILES = ("web_key", "searxng_secret")


def ensure(directory: Path) -> list[str]:
    created = []
    directory.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        path = directory / name
        if path.exists() and path.read_text("utf-8").strip():
            continue
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(secrets.token_urlsafe(32))
        created.append(name)
    return created


def main() -> int:
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "/run/aiotech")
    created = ensure(target)
    print(f"secrets créés : {', '.join(created)}" if created else "secrets déjà présents")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
