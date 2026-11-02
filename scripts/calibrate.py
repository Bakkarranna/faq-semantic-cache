#!/usr/bin/env python
"""Calibrate the cache threshold against the labelled test set.

For every paraphrase question it reports the similarity to the FAQ it is
meant to match and that FAQ's rank in the search results. For every
negative question it reports the best similarity the bank produced. The
two distributions show which thresholds separate the groups.

    python scripts/calibrate.py [model_name]
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from faqcache.config import load_settings  # noqa: E402
from faqcache.data import load_faqs, load_test_set  # noqa: E402
from faqcache.embeddings import get_embedder  # noqa: E402


def main() -> int:
    settings = load_settings()
    if len(sys.argv) > 1:
        settings = replace(settings, model_name=sys.argv[1])
    faqs = load_faqs()
    test_set = load_test_set()
    embedder = get_embedder(settings.model_name)

    faq_vectors = embedder.encode([faq.question for faq in faqs])
    id_to_index = {faq.id: index for index, faq in enumerate(faqs)}

    paraphrases: list[tuple[float, str, str, int]] = []
    negatives: list[tuple[float, str, int]] = []
    rank_one = 0
    for item in test_set:
        vector = embedder.encode([item.question])
        scores = faq_vectors @ vector[0]
        order = np.argsort(-scores)
        best = int(order[0])
        best_score = float(scores[best])
        if item.expected == "none":
            negatives.append((best_score, item.question, best))
            continue
        wanted = id_to_index[item.expected]
        wanted_score = float(scores[wanted])
        rank = int(np.where(order == wanted)[0][0]) + 1
        paraphrases.append((wanted_score, item.question, item.expected, rank))
        if rank == 1:
            rank_one += 1

    print(f"model           {settings.model_name}")
    print(f"dim             {faq_vectors.shape[1]}")
    print(f"rank-one rate   {rank_one}/{len(paraphrases)}")
    print()

    def report(name: str, values: list[float]) -> None:
        arr = np.array(sorted(values))
        marks = [0, 10, 25, 50, 75, 90, 100]
        parts = [f"p{mark}={np.percentile(arr, mark):.3f}" for mark in marks]
        print(f"{name:12s} n={len(values):3d}  " + "  ".join(parts))

    report("paraphrase", [value[0] for value in paraphrases])
    report("negative", [value[0] for value in negatives])
    print()

    for value in sorted(paraphrases)[:12]:
        score, question, expected, rank = value
        print(f"  {score:.3f} rank{rank}  {question[:50]!r} -> {expected}")
    print()
    for score, question, best in sorted(negatives, reverse=True)[:12]:
        print(f"  {score:.3f}  {question[:50]!r} -> FAQ-{best + 1:02d}")
    print()

    for threshold in (0.70, 0.75, 0.80, 0.86, 0.90):
        hits = sum(1 for value in paraphrases if value[0] >= threshold)
        false = sum(1 for value in negatives if value[0] >= threshold)
        total = len(paraphrases)
        print(
            f"threshold {threshold:.2f}: paraphrase hits {hits}/{total}, "
            f"false hits {false}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
