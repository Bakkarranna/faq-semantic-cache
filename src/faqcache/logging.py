"""Request logging.

Every question leaves one line with the fields an operator needs to debug a
cache: the question, whether it hit or missed, the similarity that produced the
decision, the latency, and the tokens an LLM call would have cost.

Nothing customer-specific is written. Questions are logged as submitted
because they are the input to tuning the threshold, while answers are stored
only as their length, so an order number a customer happens to type never
reaches the log file.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LogRecord:
    """One logged request."""

    timestamp: float
    question: str
    outcome: str
    similarity: float | None
    matched_faq_id: str | None
    latency_ms: float
    answer_chars: int
    prompt_tokens: int
    completion_tokens: int
    negations: list[str]
    stored: bool

    def to_json(self) -> str:
        return json.dumps(
            {
                "ts": round(self.timestamp, 3),
                "question": self.question,
                "outcome": self.outcome,
                "similarity": None if self.similarity is None else round(self.similarity, 4),
                "faq": self.matched_faq_id,
                "latency_ms": round(self.latency_ms, 2),
                "answer_chars": self.answer_chars,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "negations": self.negations,
                "stored": self.stored,
            },
            ensure_ascii=False,
        )


class RequestLogger:
    """An append-only JSONL logger, safe to share between threads."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.records: list[LogRecord] = []
        self._lock = threading.Lock()
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        question: str,
        outcome: str,
        similarity: float | None,
        matched_faq_id: str | None,
        latency_ms: float,
        answer: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        negations: list[str] | None = None,
        stored: bool = False,
    ) -> LogRecord:
        """Append one record and return it."""

        record = LogRecord(
            timestamp=time.time(),
            question=question,
            outcome=outcome,
            similarity=similarity,
            matched_faq_id=matched_faq_id,
            latency_ms=latency_ms,
            answer_chars=len(answer),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            negations=list(negations or []),
            stored=stored,
        )
        with self._lock:
            self.records.append(record)
            if self.path is not None:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(record.to_json() + "\n")
        return record


def summarise(records: list[LogRecord]) -> dict[str, float]:
    """Aggregate log records into the numbers the README table needs."""

    total = len(records)
    hits = sum(1 for record in records if record.outcome == "hit")
    misses = sum(1 for record in records if record.outcome == "miss")
    refused = sum(1 for record in records if record.outcome == "refused")
    llm_calls = misses + refused
    tokens = sum(record.prompt_tokens + record.completion_tokens for record in records)
    latencies = sorted(record.latency_ms for record in records)
    return {
        "total": total,
        "hits": hits,
        "misses": misses,
        "refused": refused,
        "hit_rate": (hits / total) if total else 0.0,
        "llm_calls": llm_calls,
        "tokens": tokens,
        "avg_latency_ms": (sum(latencies) / total) if total else 0.0,
        "p95_latency_ms": (
            latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))] if latencies else 0.0
        ),
    }
