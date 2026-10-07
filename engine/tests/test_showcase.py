import json

from aiotech.showcase import QUERIES, record, slug, write


async def test_showcase_records_real_events_with_guardrails(tmp_path):
    pii = "Écrivez-moi à jean.dupont@example.com : puis-je me rétracter après un achat en ligne ?"
    injection = QUERIES[-1]
    recorded = await record([QUERIES[0], pii, injection])

    events = recorded[QUERIES[0]]
    assert [e["event"] for e in events][-2:] == ["possibilities", "done"]
    assert len(events[-2]["data"]["possibilities"]) >= 2

    assert "jean.dupont" not in json.dumps(recorded[pii], ensure_ascii=False)
    assert [e["event"] for e in recorded[injection]] == ["blocked", "done"]

    paths = write(tmp_path / "demo", recorded)
    index = json.loads((tmp_path / "demo" / "index.json").read_text("utf-8"))
    assert [q["query"] for q in index["queries"]] == [QUERIES[0], pii, injection]
    assert {p.name for p in paths} == {"index.json", *(f"{slug(q)}.json" for q in recorded)}
