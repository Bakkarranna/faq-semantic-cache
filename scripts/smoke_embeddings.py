#!/usr/bin/env python
"""Sanity check for the real embedding model.

Runs in CI after ``pip install -r requirements.txt``, so it is the gate that
the model still downloads and still ranks paraphrases above other FAQs. It
needs no LLM: it only reports what the vectors say.

    python scripts/smoke_embeddings.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from faqcache.cache import HIT, SemanticCache  # noqa: E402
from faqcache.config import load_settings  # noqa: E402
from faqcache.data import label_counts, load_faqs, load_test_set  # noqa: E402
from faqcache.embeddings import get_embedder  # noqa: E402

# Gates and the operating point, in the units the measurement produced
# (results/summary.json and results/sweep.json). The threshold is read from
# the settings rather than restated here so this script and the package can
# never disagree about where the cache answers.
RECALL_FLOOR = 0.25        # of paraphrases: 0.35 measured at the threshold
FALSE_HIT_LIMIT = 0        # one wrongly served question fails the run
HIT_MEAN_MS = 250.0        # measured 39.7 ms; an LLM call took 3532 ms
HIT_MAX_MS = 1000.0        # one slow hit on a loaded machine


def main() -> int:
    settings = load_settings()
    faqs = load_faqs()
    test_set = load_test_set()
    counts = label_counts()

    started = time.perf_counter()
    embedder = get_embedder(settings.model_name)
    cache = SemanticCache(settings, embedder=embedder, client=None)
    cache.warm(faqs)
    warm_ms = (time.perf_counter() - started) * 1000.0

    vectors = embedder.encode([faq.question for faq in faqs])
    similarity = vectors @ vectors.T
    off_diagonal = similarity - 999.0 * _eye(len(faqs))
    print(f"model            {settings.model_name}")
    print(f"vector dimension {embedder.dim}")
    print(f"cached questions {len(faqs)}")
    print(f"warm-up          {warm_ms:.0f} ms for {len(faqs)} questions")
    print(f"closest pair     {off_diagonal.max():.4f} apart in cosine similarity")

    hits = language_hits = 0
    false_hits = 0
    hit_ms: list[float] = []
    for item in test_set:
        decision = cache.ask(item.question)
        if item.expected == "none" and decision.outcome == HIT:
            false_hits += 1
        if item.expected != "none" and decision.outcome == HIT:
            hits += 1
            if item.language == "hinglish":
                language_hits += 1
            hit_ms.append(decision.latency_ms)

    hinglish = sum(1 for item in test_set if item.language == "hinglish")
    paraphrases = counts["paraphrase"]
    print(f"threshold        {settings.threshold:.2f}")
    print(f"paraphrase hits  {hits}/{paraphrases}")
    print(f"hinglish hits    {language_hits}/{hinglish}")
    print(f"false hits       {false_hits}/{counts['negative']}")
    print(f"average hit      {sum(hit_ms) / len(hit_ms):.1f} ms"
          if hit_ms else "average hit      no hits")

    failures = 0
    if off_diagonal.max() >= settings.threshold:
        print("FAIL two distinct FAQs are closer than the threshold")
        failures += 1
    if hits < paraphrases * RECALL_FLOOR:
        print("FAIL the model is not ranking paraphrases highly enough")
        failures += 1
    if false_hits > FALSE_HIT_LIMIT:
        print(f"FAIL {false_hits} negative questions were served from the cache")
        failures += 1
    if not hit_ms:
        print("FAIL no question was served from the cache")
        failures += 1
    elif sum(hit_ms) / len(hit_ms) > HIT_MEAN_MS:
        print(f"FAIL the average hit took {sum(hit_ms) / len(hit_ms):.1f} ms")
        failures += 1
    elif max(hit_ms) > HIT_MAX_MS:
        print(f"FAIL a hit took {max(hit_ms):.1f} ms")
        failures += 1
    if failures:
        print(f"{failures} check(s) failed")
        return 1
    print("all checks passed")
    return 0


def _eye(size: int):
    import numpy as np

    return np.eye(size, dtype="float32")


if __name__ == "__main__":
    raise SystemExit(main())
