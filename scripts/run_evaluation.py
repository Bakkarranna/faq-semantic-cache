#!/usr/bin/env python
"""Measure the cache: baseline, threshold sweep, request log.

Every number in the README comes from this script. Run it with:

    python scripts/run_evaluation.py

The measurement runs in two phases so no number in the report is estimated.

Phase 1, the baseline: the cache is bypassed and all 100 test questions go to
the LLM. The wall-clock time spent inside each LLM call is recorded per
question, together with the token count.

Phase 2, the cache at one threshold: the cache is warmed with the 30 FAQs and
each question is asked. A hit is timed directly. A miss is charged the time of
that question's own baseline LLM call, so a miss is never optimistic.

The sweep repeats phase 2 from 0.75 to 0.95 in steps of 0.05 and records, per
threshold, the hit rate, the false hits, the number of LLM calls and the
average latency.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from faqcache.cache import SemanticCache  # noqa: E402
from faqcache.config import Settings, load_settings  # noqa: E402
from faqcache.data import load_faqs, load_test_set  # noqa: E402
from faqcache.embeddings import get_embedder  # noqa: E402
from faqcache.llm import OllamaClient, default_client  # noqa: E402
from faqcache.logging import RequestLogger, summarise  # noqa: E402

RESULTS = REPO_ROOT / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="measure the cache decision path only, with no LLM latency",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    RESULTS.mkdir(exist_ok=True)
    settings = load_settings()
    faqs = load_faqs()
    test_set = load_test_set()

    embedder = get_embedder(settings.model_name)
    client = None if args.skip_llm else default_client(settings)
    if client is None and not args.skip_llm:
        print("Ollama is not answering; measuring the decision path only.")
        args.skip_llm = True

    print(f"model: {settings.model_name}")
    print(f"llm:   {'skipped' if client is None else client.model}")
    print(f"faqs {len(faqs)}, test questions {len(test_set)}")

    baseline = measure_baseline(test_set, client, args.skip_llm)
    _write_json(RESULTS / "baseline.json", baseline)
    print(
        f"baseline: {baseline['avg_latency_ms']:.0f} ms average, "
        f"{baseline['llm_calls']} LLM calls, {baseline['tokens']} tokens"
    )

    sweep: list[dict] = []
    for threshold in settings.sweep_thresholds:
        sweep.append(
            evaluate_threshold(
                threshold, settings, faqs, test_set, embedder, baseline, args.skip_llm
            )
        )
        print(
            f"t={threshold:.2f}  hit rate {sweep[-1]['hit_rate']:.0%}  "
            f"false hits {sweep[-1]['false_hits']}/{baseline['negatives']}  "
            f"avg {sweep[-1]['avg_latency_ms']:.0f} ms  "
            f"llm calls {sweep[-1]['llm_calls']}/{sweep[-1]['total']}"
        )
    _write_json(RESULTS / "sweep.json", sweep)

    chosen = choose_threshold(sweep)
    _write_json(
        RESULTS / "summary.json",
        build_summary(settings, faqs, test_set, baseline, chosen, client),
    )
    print("\nchosen threshold", chosen["threshold"])
    print("summary written to results/summary.json")
    return 0


def measure_baseline(test_set, client: OllamaClient | None, skip_llm: bool) -> dict:
    """Latency, tokens and LLM calls with the cache bypassed.

    ``llm_ms`` is the wall-clock time inside one LLM call, without the
    embedding time, so it can be added to a measured miss later.
    """

    logger = RequestLogger(path=RESULTS / "baseline_requests.jsonl")
    _reset(logger.path)
    per_question: dict[str, float] = {}
    for question in test_set:
        if client is None:
            text, llm_ms, tokens = f"[no llm] {question.expected}. ", 0.0, 0
        else:
            started = time.perf_counter()
            completion = client.generate(question.question)
            llm_ms = (time.perf_counter() - started) * 1000.0
            text = completion.text
            tokens = completion.prompt_tokens + completion.completion_tokens
        per_question[question.question] = llm_ms
        logger.log(
            question=question.question,
            outcome="baseline",
            similarity=None,
            matched_faq_id=None,
            latency_ms=llm_ms,
            answer=text,
            prompt_tokens=tokens,
        )
    summary = summarise(logger.records)
    summary["negatives"] = sum(1 for q in test_set if q.expected == "none")
    summary["per_question_llm_ms"] = per_question
    summary["skip_llm"] = skip_llm
    return summary


def evaluate_threshold(threshold, settings, faqs, test_set, embedder, baseline, skip_llm) -> dict:
    """Ask every question at one threshold and score the decisions."""

    settings = Settings(
        threshold=threshold,
        ttl_days=settings.ttl_days,
        model_name=settings.model_name,
        ollama_model=settings.ollama_model,
        ollama_url=settings.ollama_url,
        negation_words=settings.negation_words,
        user_data_patterns=settings.user_data_patterns,
    )
    cache = SemanticCache(settings, embedder=embedder, client=None)
    cache.warm(faqs)
    logger = RequestLogger(path=RESULTS / f"requests_t{threshold:.2f}.jsonl")
    _reset(logger.path)

    llm_ms = baseline["per_question_llm_ms"]
    hits = misses = false_hits = wrong_faq = _ = 0
    hit_latencies: list[float] = []
    question_latencies: list[float] = []
    llm_calls = 0

    for question in test_set:
        decision = cache.ask(question.question)
        if decision.outcome == "hit":
            hits += 1
            hit_latencies.append(decision.latency_ms)
            question_latencies.append(decision.latency_ms)
            if question.expected == "none":
                false_hits += 1
            elif decision.matched_faq_id != question.expected:
                wrong_faq += 1
        else:
            misses += 1
            llm_calls += 1
            answer_ms = llm_ms.get(question.question, baseline["avg_latency_ms"])
            question_latencies.append(decision.latency_ms + answer_ms)
        logger.log(
            question=question.question,
            outcome=decision.outcome,
            similarity=decision.similarity,
            matched_faq_id=decision.matched_faq_id,
            latency_ms=decision.latency_ms,
            answer=decision.answer,
            negations=decision.negations,
            stored=decision.stored,
        )

    total = len(test_set)
    negatives = sum(1 for q in test_set if q.expected == "none")
    return {
        "threshold": threshold,
        "total": total,
        "negatives": negatives,
        "hits": hits,
        "misses": misses,
        "hit_rate": hits / total,
        "false_hits": false_hits,
        "false_hit_rate": false_hits / negatives if negatives else 0.0,
        "wrong_faq_served": wrong_faq,
        "avg_hit_latency_ms": statistics.fmean(hit_latencies) if hit_latencies else 0.0,
        "avg_latency_ms": statistics.fmean(question_latencies) if question_latencies else 0.0,
        "llm_calls": llm_calls,
        "llm_call_reduction": 1.0 - llm_calls / total,
        "llm_ms": "skipped" if skip_llm else None,
    }


def choose_threshold(sweep: list[dict]) -> dict:
    """Pick the highest hit rate among the thresholds with no false hits.

    The brief asks for the best hit rate with false hits near zero, so a
    threshold is eligible only when it served no negative question from the
    cache.
    """

    clean = [row for row in sweep if row["false_hits"] == 0]
    pool = clean or sweep
    return max(pool, key=lambda row: (row["hit_rate"], -row["false_hits"], row["threshold"]))


def build_summary(settings, faqs, test_set, baseline, chosen, client) -> dict:
    """Collect the numbers the README quotes, with their provenance."""

    speedup = (
        baseline["avg_latency_ms"] / chosen["avg_latency_ms"]
        if chosen["avg_latency_ms"]
        else 0.0
    )
    return {
        "threshold": chosen["threshold"],
        "embedding_model": settings.model_name,
        "llm_model": "skipped" if client is None else client.model,
        "llm_ms_measured": not baseline["skip_llm"],
        "faqs_cached": len(faqs),
        "test_questions": chosen["total"],
        "paraphrases": sum(1 for q in test_set if q.expected != "none"),
        "negatives": chosen["negatives"],
        "hit_rate": chosen["hit_rate"],
        "hits": chosen["hits"],
        "false_hits": chosen["false_hits"],
        "wrong_faq_served": chosen["wrong_faq_served"],
        "avg_hit_latency_ms": chosen["avg_hit_latency_ms"],
        "avg_latency_ms": chosen["avg_latency_ms"],
        "baseline_avg_latency_ms": baseline["avg_latency_ms"],
        "baseline_llm_calls": baseline["llm_calls"],
        "cached_llm_calls": chosen["llm_calls"],
        "llm_call_reduction": chosen["llm_call_reduction"],
        "latency_speedup": speedup,
        "measured_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _reset(path: Path | None) -> None:
    """Remove a log so a rerun starts from an empty file."""

    if path is not None and path.exists():
        path.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
