"""Request logger tests."""

from __future__ import annotations

import json

import pytest

from faqcache.logging import LogRecord, RequestLogger, summarise


def test_a_record_is_written_as_one_json_line(tmp_path) -> None:
    path = tmp_path / "requests.jsonl"
    logger = RequestLogger(path)
    logger.log(
        question="refund kab milega",
        outcome="hit",
        similarity=0.9213,
        matched_faq_id="REFUND-01",
        latency_ms=4.17,
        answer="Refunds reach the source account in 5 to 7 working days.",
        negations=[],
        stored=False,
    )
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["outcome"] == "hit"


def test_the_answer_itself_is_never_written_to_the_log(tmp_path) -> None:
    path = tmp_path / "requests.jsonl"
    logger = RequestLogger(path)
    answer = "Your order ORD392104 is out for delivery."
    logger.log(
        question="where is my order",
        outcome="miss",
        similarity=None,
        matched_faq_id=None,
        latency_ms=1200.0,
        answer=answer,
    )
    written = path.read_text(encoding="utf-8")
    assert "out for delivery" not in written  # the answer is never written
    assert json.loads(written)["answer_chars"] == len(answer)


def test_the_similarity_is_rounded_in_the_file_but_kept_on_the_record() -> None:
    logger = RequestLogger(None)
    record = logger.log(
        question="q",
        outcome="hit",
        similarity=0.9123456789,
        matched_faq_id="F1",
        latency_ms=1.0,
        answer="a",
    )
    assert record.similarity == 0.9123456789
    assert json.loads(record.to_json())["similarity"] == 0.9123


def _record(outcome: str, latency_ms: float, prompt_tokens: int = 0, completion_tokens: int = 0):
    return LogRecord(
        timestamp=0.0,
        question="q",
        outcome=outcome,
        similarity=0.9,
        matched_faq_id="F1",
        latency_ms=latency_ms,
        answer_chars=10,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        negations=[],
        stored=False,
    )


def test_summarise_counts_hits_misses_and_refusals() -> None:
    summary = summarise(
        [
            _record("hit", 4.0),
            _record("hit", 6.0),
            _record("miss", 900.0, prompt_tokens=30, completion_tokens=20),
            _record("refused", 800.0, prompt_tokens=10, completion_tokens=10),
        ]
    )
    assert summary["total"] == 4
    assert summary["hits"] == 2
    assert summary["misses"] == 1
    assert summary["refused"] == 1
    assert summary["llm_calls"] == 2
    assert summary["hit_rate"] == 0.5
    assert summary["tokens"] == 70
    assert summary["avg_latency_ms"] == pytest.approx((4 + 6 + 900 + 800) / 4)
    assert summary["p95_latency_ms"] == 900.0


def test_summarise_of_nothing_is_zero_rather_than_an_error() -> None:
    assert summarise([])["hit_rate"] == 0.0
    assert summarise([])["avg_latency_ms"] == 0.0
