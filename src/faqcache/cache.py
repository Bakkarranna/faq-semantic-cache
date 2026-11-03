"""The cache policy: embed, look up, serve, or call the LLM.

The decision order is deliberate:

1. skip the cache when the question carries a negation word;
2. look up the nearest cached question and serve it when the similarity passes
   the threshold and the entry has not expired;
3. otherwise call the LLM, and store the new question and answer when the
   answer is not specific to one customer.

Steps 1 and 3 are what keep the hit rate honest, and they are the two places a
production cache needs a policy of its own rather than the default behaviour.

A question declined by the negation guard still reaches the LLM: the guard
withholds a cached answer, not an answer. What it withholds is the cache, so a
question that negates an FAQ never reads a sentence written for its opposite.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from faqcache.data import Faq
from faqcache.embeddings import get_embedder
from faqcache.llm import Completion
from faqcache.negation import find_negations, looks_like_user_data
from faqcache.store import CacheEntry, Match, VectorStore

HIT = "hit"
MISS = "miss"
REFUSED = "refused"

# A cached answer older than this is recomputed even though it still matches.
SECONDS_PER_DAY = 86_400


@dataclass(frozen=True)
class CacheDecision:
    """The outcome of one question, with everything the report needs."""

    outcome: str
    answer: str
    similarity: float | None
    matched_faq_id: str | None
    latency_ms: float
    negations: list[str] = field(default_factory=list)
    completion: Completion | None = None
    stored: bool = False
    expired: bool = False


class SemanticCache:
    """A semantic cache in front of an LLM, backed by a FAISS index."""

    def __init__(self, settings, embedder=None, client=None, clock=time.monotonic) -> None:
        """Wire the cache to an embedder and to an LLM, when there is one.

        ``client=None`` means no LLM is attached: a miss then falls back to
        the notice in ``_answer_via_llm`` instead of spending tokens. The
        threshold sweep and the smoke test run that way because they
        measure decisions, not answers, so a machine with no Ollama server
        gives the same numbers as one that has it.
        """

        self.settings = settings
        self._clock = clock
        self._embedder = embedder or get_embedder(settings.model_name)
        self._client = client
        self._store = VectorStore(dim=self._embedder.dim)

    def __len__(self) -> int:
        return len(self._store)

    def add_faqs(self, faqs: list[Faq]) -> None:
        """Populate the cache from the FAQ set, one vector per question."""

        vectors = self._embedder.encode([faq.question for faq in faqs])
        now = self._clock()
        for faq, vector in zip(faqs, vectors, strict=True):
            self._store.add(
                vector,
                CacheEntry(
                    faq_id=faq.id,
                    question=faq.question,
                    answer=faq.answer,
                    volatile=faq.volatile,
                    stored_at=now,
                ),
            )

    def warm(self, faqs: list[Faq]) -> None:
        """Alias used by the evaluation harness for readability."""

        self.add_faqs(faqs)

    def ask(self, question: str) -> CacheDecision:
        """Answer one question through the cache."""

        started = time.monotonic()
        negations = find_negations(question, self.settings.negation_words)
        if negations:
            answer, completion = self._answer_via_llm(question)
            return CacheDecision(
                outcome=REFUSED,
                answer=answer,
                similarity=None,
                matched_faq_id=None,
                latency_ms=_elapsed_ms(started),
                negations=negations,
                completion=completion,
            )

        vector = self._embedder.encode([question])[0]
        match = self._best_match(vector)
        if match is not None:
            expired = self._is_expired(match.entry)
            if match.similarity >= self.settings.threshold and not expired:
                return CacheDecision(
                    outcome=HIT,
                    answer=match.entry.answer,
                    similarity=match.similarity,
                    matched_faq_id=match.entry.faq_id,
                    latency_ms=_elapsed_ms(started),
                    expired=False,
                )

        answer, completion = self._answer_via_llm(question)
        stored = self._store_answer(question, answer, match)
        return CacheDecision(
            outcome=MISS,
            answer=answer,
            similarity=match.similarity if match else None,
            matched_faq_id=None,
            latency_ms=_elapsed_ms(started),
            completion=completion,
            stored=stored,
        )

    # Internals ---------------------------------------------------------------

    def _answer_via_llm(self, question: str) -> tuple[str, Completion | None]:
        """Answer a question with the LLM, or with a notice when it is absent.

        Returning the completion alongside the text keeps the token count
        available to the caller that paid for it. With no client attached the
        text is a notice and there is no completion to report.
        """

        if self._client is None:
            return (
                (
                    "Support is unavailable right now. Please open the Orders "
                    "page or contact support@example.com."
                ),
                None,
            )
        completion = self._client.generate(question)
        return completion.text, completion

    def _best_match(self, vector) -> Match | None:
        matches = self._store.search(vector, k=1)
        return matches[0] if matches else None

    def _is_expired(self, entry: CacheEntry) -> bool:
        """Report whether a volatile answer has outlived its TTL."""

        if not entry.volatile or self.settings.ttl_days <= 0:
            return False
        return (self._clock() - entry.stored_at) > self.settings.ttl_days * SECONDS_PER_DAY

    def _store_answer(self, question: str, answer: str, match: Match | None) -> bool:
        """Store a new answer unless it is specific to one customer.

        A miss against a known FAQ also refreshes the stored answer, which is
        how an expired or reworded answer gets replaced.
        """

        if looks_like_user_data(answer, self.settings.user_data_patterns):
            return False
        matched_above_threshold = (
            match is not None and match.similarity >= self.settings.threshold
        )
        source_faq = match.entry if matched_above_threshold else None
        entry = CacheEntry(
            faq_id=source_faq.faq_id if source_faq else f"LLM-{len(self._store) + 1:03d}",
            question=question,
            answer=answer,
            volatile=bool(source_faq.volatile) if source_faq else False,
            stored_at=self._clock(),
        )
        if source_faq is not None:
            self._replace(entry)
            return True
        self._store.add(self._embedder.encode([question])[0], entry)
        return True

    def _replace(self, entry: CacheEntry) -> None:
        """Swap the answer of the entry a question matched, keeping its vector."""

        for position, existing in enumerate(self._store.entries):
            if existing.faq_id == entry.faq_id:
                self._store.remove_at(position)
                self._store.add(self._embedder.encode([entry.question])[0], entry)
                return


def _elapsed_ms(started: float) -> float:
    return (time.monotonic() - started) * 1000.0
