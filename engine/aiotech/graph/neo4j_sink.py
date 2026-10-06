"""
Persistance du graphe de connaissances dans Neo4j (service interne du docker-compose).

Chaque recherche écrit ses entités, affirmations, sources et contradictions :
    (:Search)-[:FOUND]->(:Claim)-[:ABOUT]->(:Entity)
    (:Claim)-[:FROM]->(:Source)
    (:Search)-[:CONTRADICTION {attribute, resolved, winner}]->(:Entity)
L'écriture est faite après la réponse et ne bloque jamais la recherche : un échec devient
un avertissement. La requête stockée est la version masquée (aucune donnée personnelle).
"""
from __future__ import annotations

from typing import Any

from neo4j import AsyncDriver, AsyncGraphDatabase, AsyncManagedTransaction

_STATEMENTS: tuple[str, ...] = (
    "MERGE (s:Search {id: $search_id}) SET s.query = $query, s.at = datetime()",
    "UNWIND $entities AS e MERGE (n:Entity {key: e.key}) SET n.label = e.label",
    "UNWIND $sources AS src MERGE (so:Source {id: src.id}) "
    "SET so.title = src.title, so.url = src.url, so.origin = src.origin, so.reliability = src.reliability",
    "MATCH (s:Search {id: $search_id}) UNWIND $claims AS c "
    "MERGE (cl:Claim {id: c.id}) SET cl.attribute = c.attribute, cl.value = c.value, cl.kind = c.kind, "
    "cl.comparator = c.comparator, cl.extractor = c.extractor MERGE (s)-[:FOUND]->(cl) "
    "WITH cl, c MATCH (n:Entity {key: c.entity}) MERGE (cl)-[:ABOUT]->(n) "
    "WITH cl, c MATCH (so:Source {id: c.source}) MERGE (cl)-[:FROM]->(so)",
    "MATCH (s:Search {id: $search_id}) UNWIND $contradictions AS k MATCH (n:Entity {key: k.entity_key}) "
    "MERGE (s)-[r:CONTRADICTION {attribute: k.attribute}]->(n) SET r.resolved = k.resolved, r.winner = k.winner",
)


class Neo4jSink:
    def __init__(self, url: str, user: str, password: str) -> None:
        self._driver: AsyncDriver = AsyncGraphDatabase.driver(url, auth=(user, password))

    async def write(self, search_id: str, query: str, graph: dict[str, Any]) -> None:
        params = {
            "search_id": search_id,
            "query": query,
            "entities": [{"key": e["key"], "label": e["label"]} for e in graph.get("entities", [])],
            "sources": [
                {"id": s["id"], "title": s["title"], "url": s.get("url"), "origin": s.get("origin"),
                 "reliability": s.get("reliability")}
                for s in graph.get("sources", [])
            ],
            "claims": [
                {k: c[k] for k in ("id", "attribute", "value", "kind", "comparator", "extractor", "entity", "source")}
                for c in graph.get("claims", [])
            ],
            "contradictions": [
                {"entity_key": k["entity_key"], "attribute": k["attribute"], "resolved": k["resolved"],
                 "winner": k.get("winner")}
                for k in graph.get("contradictions", [])
            ],
        }

        async def work(tx: AsyncManagedTransaction) -> None:
            for statement in _STATEMENTS:
                result = await tx.run(statement, params)
                await result.consume()

        async with self._driver.session() as session:
            await session.execute_write(work)

    async def ping(self) -> bool:
        try:
            await self._driver.verify_connectivity()
        except Exception:
            return False
        return True

    async def close(self) -> None:
        await self._driver.close()
