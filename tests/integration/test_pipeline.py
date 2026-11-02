"""End-to-end tests with the real embedding model.

These download ``paraphrase-multilingual-MiniLM-L12-v2`` on the first run, so
they are kept apart from the unit tests, which run against the hashing
embedder and need no model. The LLM is not contacted here: the questions used
below are either hits or refusals, both of which never reach the model.
"""

from __future__ import annotations

import pytest

from faqcache.cache import HIT, MISS, REFUSED, SemanticCache
from faqcache.config import load_settings
from faqcache.data import load_faqs, load_test_set
from faqcache.embeddings import SentenceTransformerEmbedder, get_embedder


@pytest.fixture(scope="module")
def model() -> SentenceTransformerEmbedder:
    return get_embedder()


@pytest.fixture(scope="module")
def cache(model) -> SemanticCache:
    settings = load_settings()
    cache = SemanticCache(settings, embedder=model, client=None)
    cache.warm(load_faqs())
    return cache


def test_the_model_downloads_and_normalises(model) -> None:
    assert model.dim == 384
    vectors = model.encode(["kaise ho", "how are you"])
    assert vectors.shape == (2, 384)
    norms = (vectors ** 2).sum(axis=1) ** 0.5
    assert norms == pytest.approx([1.0, 1.0], abs=1e-5)


@pytest.mark.parametrize(
    "question",
    [
        "How long will delivery take?",
        "refund kab milega",
        "mera parcel kahan hai",
    ],
)
def test_a_paraphrase_is_served_from_the_cache(cache, question: str) -> None:
    decision = cache.ask(question)
    assert decision.outcome == HIT
    assert decision.similarity >= 0.86
    assert decision.matched_faq_id is not None
    assert decision.latency_ms > 0.0


def test_an_out_of_scope_question_is_a_miss(cache) -> None:
    decision = cache.ask("do you sell second hand washing machines in Indore")
    assert decision.outcome == MISS
    assert decision.similarity < 0.86


@pytest.mark.parametrize(
    "question,word",
    [
        ("refund nahi milega kya", "nahi"),
        ("mujhe refund nahi chahiye", "nahi"),
        ("I do not want to return this, can I exchange instead", "not"),
    ],
)
def test_a_negated_question_is_never_served_from_the_cache(cache, question: str, word: str) -> None:
    decision = cache.ask(question)
    assert decision.outcome == REFUSED
    assert word in decision.negations
    assert decision.similarity is None


def test_every_paraphrase_in_the_test_set_is_measured(cache) -> None:
    """Report the hit rate rather than assert a number that may drift."""

    test_set = load_test_set()
    hits = sum(
        1
        for item in test_set
        if item.expected != "none" and cache.ask(item.question).outcome == HIT
    )
    paraphrases = sum(1 for item in test_set if item.expected != "none")
    print(f"\nparaphrase hits: {hits}/{paraphrases}")
    assert hits >= paraphrases * 0.8
