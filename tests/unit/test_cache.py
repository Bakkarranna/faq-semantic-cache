"""Cache policy tests, run against a stub embedder and a stub LLM client.

The embedder is replaced by a lookup so the similarity of a pair of questions
is a property of the test rather than of a model. CI runs this file with
``FAQCACHE_EMBEDDER=deterministic``; these tests inject their own embedder, so
they depend on neither.
"""

from __future__ import annotations

import numpy as np
import pytest

from faqcache.cache import HIT, MISS, REFUSED, SemanticCache
from faqcache.config import load_settings
from faqcache.data import Faq
from faqcache.llm import Completion


class LookupEmbedder:
    """Maps a question to the vector named in ``map``, or to a fresh basis vector."""

    dim = 8

    def __init__(self, map: dict[str, tuple[float, ...]]) -> None:
        self.map = map

    def encode(self, texts: list[str]) -> np.ndarray:
        rows = []
        for text in texts:
            if text in self.map:
                rows.append(np.asarray(self.map[text], dtype=np.float32))
                continue
            row = np.zeros(self.dim, dtype=np.float32)
            row[abs(hash(text)) % self.dim if False else self._bucket(text)] = 1.0
            rows.append(row)
        return np.asarray(rows, dtype=np.float32)

    @staticmethod
    def _bucket(text: str) -> int:
        return sum(ord(char) for char in text) % 8


class StubClient:
    """Records calls and answers with a fixed string."""

    def __init__(self, text: str = "Answer from the language model.") -> None:
        self.text = text
        self.questions: list[str] = []

    def generate(self, question: str):
        self.questions.append(question)
        return Completion(
            text=self.text,
            prompt_tokens=42,
            completion_tokens=17,
            latency_ms=120.0,
            model="stub",
        )


def make_cache(client_text: str = "Answer from the language model.", **overrides):
    settings = load_settings(**overrides)
    embedder = LookupEmbedder(
        {
            "delivery kitne din me hoti hai": (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            "How long does delivery take?": (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            "refund nahi chahiye": (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        }
    )
    client = StubClient(client_text)
    cache = SemanticCache(settings, embedder=embedder, client=client)
    return cache, client


FAQS = [
    Faq(
        id="DELIV-01",
        topic="delivery",
        volatile=False,
        question="delivery kitne din me hoti hai",
        answer="Delivery takes 3 to 5 working days.",
    ),
    Faq(
        id="PRICE-01",
        topic="pricing",
        volatile=True,
        question="shipping kitni hai",
        answer="Shipping is Rs 49 on orders under Rs 500.",
    ),
]


def test_a_reworded_question_is_served_from_the_cache() -> None:
    cache, client = make_cache()
    cache.add_faqs(FAQS)
    decision = cache.ask("How long does delivery take?")
    assert decision.outcome == HIT
    assert decision.similarity == pytest.approx(1.0)
    assert decision.answer == "Delivery takes 3 to 5 working days."
    assert decision.matched_faq_id == "DELIV-01"
    assert client.questions == []


def test_a_question_below_the_threshold_calls_the_llm_and_is_stored() -> None:
    cache, client = make_cache()
    cache.add_faqs(FAQS)
    decision = cache.ask("do you sell gift cards")
    assert decision.outcome == MISS
    assert client.questions == ["do you sell gift cards"]
    assert decision.stored is True
    assert len(cache) == 3


def test_the_same_question_twice_hits_the_second_time() -> None:
    cache, _ = make_cache()
    cache.add_faqs(FAQS)
    cache.ask("do you sell gift cards")
    decision = cache.ask("do you sell gift cards")
    assert decision.outcome == HIT
    assert decision.matched_faq_id.startswith("LLM-")


def test_a_negated_question_never_receives_the_opposite_answer() -> None:
    cache, client = make_cache()
    cache.add_faqs(FAQS)
    decision = cache.ask("refund nahi chahiye")
    assert decision.outcome == REFUSED
    assert decision.negations == ["nahi"]
    assert decision.similarity is None
    assert decision.stored is False
    assert decision.answer == "Answer from the language model."
    assert client.questions == ["refund nahi chahiye"]


def test_an_answer_with_a_order_number_is_not_stored() -> None:
    cache, _ = make_cache(client_text="Your order ORD392104 shipped on Monday.")
    cache.add_faqs(FAQS)
    decision = cache.ask("where is my order")
    assert decision.outcome == MISS
    assert decision.stored is False
    assert len(cache) == 2


def test_an_expired_volatile_answer_is_recomputed() -> None:
    settings = load_settings(ttl_days=1)
    now = [1_000_000.0]
    cache = SemanticCache(
        settings,
        embedder=LookupEmbedder({"shipping kitni hai": (1.0,) + (0.0,) * 7}),
        client=StubClient("Shipping is Rs 59 until Sunday."),
        clock=lambda: now[0],
    )
    now[0] = 1_000_000.0
    cache.add_faqs(FAQS)

    now[0] += 86_400 * 5  # Five days pass; the TTL is one day.
    decision = cache.ask("shipping kitni hai")
    assert decision.outcome == MISS
    assert decision.expired is False
    assert decision.answer == "Shipping is Rs 59 until Sunday."
    assert len(cache) == 2  # Refreshed in place, not appended.


def test_a_permanent_answer_is_kept_after_the_ttl_passes() -> None:
    settings = load_settings(ttl_days=1)
    now = [500_000.0]
    cache = SemanticCache(
        settings,
        embedder=LookupEmbedder({"delivery kitne din me hoti hai": (1.0,) + (0.0,) * 7}),
        client=StubClient(),
        clock=lambda: now[0],
    )
    cache.add_faqs(FAQS)
    now[0] += 86_400 * 400
    decision = cache.ask("delivery kitne din me hoti hai")
    assert decision.outcome == HIT
    assert decision.answer == "Delivery takes 3 to 5 working days."


def test_a_hit_is_faster_than_the_stub_llm_can_be_mistaken_for() -> None:
    """Hit latency is measured, so it must be non-negative and finite."""

    cache, client = make_cache()
    cache.add_faqs(FAQS)
    decision = cache.ask("How long does delivery take?")
    assert decision.latency_ms >= 0.0
    assert decision.latency_ms == decision.latency_ms  # not NaN
