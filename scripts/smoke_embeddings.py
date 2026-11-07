#!/usr/bin/env python
"""Sanity check for the real embedding model.

Runs in CI after ``pip install -r requirements.txt``, so it is the gate that
the model still downloads and still ranks paraphrases above other FAQs, and that a hit
serves the FAQ the question was paraphrased from. It
needs no LLM: it only reports what the vectors say.

The bounds below are the ones the 1,000-question measurement in results/
satisfied at the operating point: no grid threshold on the 400 negatives
produced zero false hits, so the stated false-hit budget in
scripts/run_evaluation.py, 1% of negatives, is what the gate enforces.

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
RECALL_FLOOR = 0.25        # of paraphrases: 194 of 600 measured at the threshold
FALSE_HIT_RATE_LIMIT = 0.01  # of negatives: 2 of 400 measured, budget of 1%
WRONG_HIT_RATE_LIMIT = 0.05  # of hits: 5 of 194 measured
HIT_MEAN_MS = 250.0        # measured 34.7 ms locally, 17.7 ms in CI
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

    hits = language_hits = wrong_hits = 0
    false_hits = 0
    hit_ms: list[float] = []
    for item in test_set:
        decision = cache.ask(item.question)
        if item.expected == "none" and decision.outcome == HIT:
            false_hits += 1
        if item.expected != "none" and decision.outcome == HIT:
            hits += 1
            if decision.matched_faq_id != item.expected:
                wrong_hits += 1
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

    negatives = counts["negative"]
    false_hit_rate = false_hits / negatives if negatives else 0.0
    wrong_hit_rate = wrong_hits / hits if hits else 0.0
    print(f"wrong hits       {wrong_hits}/{hits} served a different FAQ "
          f"({wrong_hit_rate * 100:.1f}% of hits)")
    print(f"false hit rate   {false_hit_rate * 100:.2f}% of {negatives} negatives")

    failures = 0
    if wrong_hit_rate > WRONG_HIT_RATE_LIMIT:
        print(f"FAIL {wrong_hits} of {hits} hits served the wrong FAQ, "
              f"above the {WRONG_HIT_RATE_LIMIT * 100:.0f}% limit")
        failures += 1
    if hits < paraphrases * RECALL_FLOOR:
        print("FAIL the model is not ranking paraphrases highly enough")
        failures += 1
    if false_hit_rate > FALSE_HIT_RATE_LIMIT:
        print(f"FAIL {false_hits} of {negatives} negative questions were served "
              f"from the cache, above the {FALSE_HIT_RATE_LIMIT * 100:.0f}% budget")
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
