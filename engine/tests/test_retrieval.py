from __future__ import annotations

from aiotech.core.feedback import ClickFeedback
from aiotech.models import Document
from aiotech.pipeline import SearchEngine
from aiotech.retrieval.corpus import CorpusStore
from aiotech.retrieval.links import TitleLinks, title_key
from aiotech.storage.kv import MemoryKV


def _doc(doc_id: str, title: str, text: str) -> Document:
    return Document(id=doc_id, title=title, text=f"{title}. {text}", reliability=0.7)


BRIDGE = [
    _doc("film", "Doctor Strange (2016 film)",
         "Doctor Strange is a superhero film directed by Scott Derrickson and released in 2016."),
    _doc("director", "Scott Derrickson",
         "Scott Derrickson is an American filmmaker born in Denver, Colorado."),
    _doc("other", "Ed Wood (film)", "Ed Wood is a biographical film about a cult director of low-budget movies."),
    _doc("noise", "Colorado River", "The river flows through canyons and deserts of the American southwest."),
]


def test_title_key_drops_trailing_qualifier_and_empty_titles() -> None:
    assert title_key("Ed Wood (film)") == ("ed", "wood")
    assert title_key("Élysée") == ("elysee",)
    assert title_key("The") == ()
    assert title_key("Of") == ()


def test_title_links_find_mentions_in_order() -> None:
    links = TitleLinks()
    links.build(BRIDGE)
    text = "A film by SCOTT DERRICKSON, unlike Ed Wood, was shot near the Colorado River."
    assert links.mentioned(text) == ["director", "other", "noise"]
    assert links.mentioned("Scott alone is not a title") == []


def test_store_exposes_links_and_first_passage() -> None:
    store = CorpusStore()
    store.add_many(BRIDGE, persist=False)
    film = store.first_passage("film")
    assert film is not None and film.document_id == "film"
    assert store.first_passage("absent") is None
    assert store.linked_documents(film.text) == ["film", "director"]


def test_fusion_scores_are_relative_to_the_best_passage() -> None:
    store = CorpusStore()
    store.add_many(BRIDGE, persist=False)
    hits = store.search("Doctor Strange superhero film", 4)
    assert hits[0].passage.document_id == "film"
    assert hits[0].score == 1.0
    assert all(0.0 <= h.score <= 1.0 for h in hits)
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


async def test_link_hop_reaches_the_bridge_document() -> None:
    store = CorpusStore()
    store.add_many(BRIDGE, persist=False)
    kv = MemoryKV()
    engine = SearchEngine(corpus=store, feedback=ClickFeedback(kv), cache=kv)
    question = "Where was the director of the 2016 superhero film Doctor Strange born?"
    lexical = [sp.passage.document_id for sp in store.search(question, 2)]
    assert "director" not in lexical
    passages, _ = engine.evidence(question, top_k=2)
    assert [sp.passage.document_id for sp in passages] == ["film", "director"]
