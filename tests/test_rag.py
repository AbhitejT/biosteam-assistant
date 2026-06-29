"""Tests for the RAG retrieval layer (no API key required)."""
from biosteam_ai.rag import Retriever, get_retriever


def test_corpus_is_built():
    r = get_retriever()
    assert len(r.documents) > 10


def test_singleton_is_reused():
    assert get_retriever() is get_retriever()


def test_fermentation_query_finds_background():
    hits = get_retriever().search("what is fermentation efficiency", k=3)
    titles = " ".join(h["title"].lower() for h in hits)
    assert "fermentation" in titles


def test_mesp_query_finds_glossary():
    hits = get_retriever().search("minimum ethanol selling price meaning", k=3)
    assert any("mesp" in h["title"].lower() or "selling price" in h["title"].lower() for h in hits)


def test_registry_docs_present():
    hits = get_retriever().search("glucose to ethanol conversion parameter range", k=5)
    assert any(h["source"] == "model registry" for h in hits)


def test_irrelevant_query_returns_nothing():
    hits = get_retriever().search("zzzqwxv nonsense token", k=4)
    assert hits == []


def test_results_sorted_by_score():
    hits = get_retriever().search("techno-economic analysis", k=4)
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True)
