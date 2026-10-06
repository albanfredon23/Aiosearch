"""
Serveur MCP d'AIOTECH Search : tout client compatible (Claude Desktop, Claude Code, autres
agents) appelle la recherche vérifiée comme un outil.

    python -m aiotech.mcp_server            transport stdio (processus local)
    python -m aiotech.mcp_server --http     HTTP streamable sur 127.0.0.1:8765 (sans clé : local)
L'API HTTP principale monte aussi ce serveur sur /mcp, protégé par la clé d'API.

Outils : aiotech_search, calculate (calculatrice confinée), read_file (lecture emprisonnée,
seulement si AIOTECH_DOCS_DIR est configuré). Aucun outil d'exécution de code.
"""
from __future__ import annotations

import argparse
import asyncio
from typing import Any

from mcp.server.mcpserver import MCPServer

from aiotech.core.adaptive import parse_depth
from aiotech.gateway.tools import ToolError
from aiotech.pipeline import SearchOptions
from aiotech.runtime import Runtime, build_runtime, summarize

INSTRUCTIONS = (
    "AIOTECH Search répond par 2 ou 3 réponses classées par fiabilité ; chaque affirmation porte un statut "
    "[FAIT], [INFÉRENCE], [INCERTAIN] ou [NON VÉRIFIÉ] et ses sources. Cite les statuts tels quels."
)


def build_mcp(runtime: Runtime) -> MCPServer:
    server = MCPServer(name="aiotech-search", instructions=INSTRUCTIONS)

    @server.tool(name="aiotech_search", description=runtime.tools.tools["aiotech_search"].description)
    async def aiotech_search(query: str, depth: str = "auto", use_web: bool = True) -> dict[str, Any]:
        options = SearchOptions(depth=parse_depth(depth), use_web=use_web)
        return summarize(await runtime.engine.search(query, options))

    @server.tool(name="calculate", description=runtime.tools.tools["calculate"].description)
    async def calculate(expression: str) -> dict[str, Any]:
        try:
            return await runtime.tools.call("calculate", {"expression": expression})
        except ToolError as exc:
            return {"error": str(exc)}

    if "read_file" in runtime.tools.tools:
        @server.tool(name="read_file", description=runtime.tools.tools["read_file"].description)
        async def read_file(path: str) -> dict[str, Any]:
            try:
                return await runtime.tools.call("read_file", {"path": path})
            except ToolError as exc:
                return {"error": str(exc)}

    return server


async def _main(http: bool, host: str, port: int) -> None:
    runtime = await build_runtime()
    try:
        server = build_mcp(runtime)
        if http:
            await server.run_streamable_http_async(host=host, port=port)
        else:
            await server.run_stdio_async()
    finally:
        await runtime.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serveur MCP AIOTECH Search")
    parser.add_argument("--http", action="store_true", help="HTTP streamable au lieu de stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    asyncio.run(_main(args.http, args.host, args.port))


if __name__ == "__main__":
    main()
