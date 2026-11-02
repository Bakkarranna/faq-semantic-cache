"""Runtime settings.

Everything the experiment depends on lives here so a run can be reproduced from
the repository alone. The defaults were used for the numbers published in
README.md.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    """One immutable snapshot of the configuration."""

    threshold: float = 0.86
    ttl_days: int = 30
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ollama_model: str = "qwen2.5:1.5b"
    ollama_url: str = "http://127.0.0.1:11434"
    max_cached_answer_chars: int = 2400
    # Words that flip the meaning of a question. A question that carries one of
    # them is never served from a cached answer written for its opposite.
    negation_words: tuple[str, ...] = (
        "not",
        "no",
        "never",
        "nahi",
        "mat",
        "nako",
        "na",
        "without",
        "cannot",
        "can't",
        "won't",
        "don't",
    )
    # Patterns that mark an answer as user-specific. Such answers are written
    # to the request log but never stored as cache entries.
    user_data_patterns: tuple[str, ...] = (
        r"\b[A-Z]{2,}[0-9]{4,}\b",           # order numbers such as ORD392104
        r"\bORD[0-9]{4,}\b",
        r"\b\d{6}\b",                        # one-time passwords
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",  # email addresses
        r"\b(?:\+91|0)?[6-9]\d{9}\b",        # Indian mobile numbers
        # Customer or account numbers, written either as "customer id 48201933"
        # or "customer account number 48201933".
        r"\bcustomer\s+(?:id|account)(?:\s+(?:no|number))?\s*[:#]?\s*\d{4,}\b",
    )
    # Thresholds measured for the trade-off table. The calibration run
    # (results/calibration.txt) placed the hardest negative at 0.681 and the
    # paraphrase median at 0.574, so the decision boundary between a safe
    # hit and a false hit lies between 0.65 and 0.70. The grid brackets that
    # boundary instead of starting above it.
    sweep_thresholds: tuple[float, ...] = field(
        default_factory=lambda: (0.60, 0.65, 0.70, 0.75, 0.80)
    )

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must lie between 0 and 1")
        if self.ttl_days < 0:
            raise ValueError("ttl_days must not be negative")


def load_settings(**overrides: object) -> Settings:
    """Build Settings from environment overrides.

    ``FAQCACHE_THRESHOLD``, ``FAQCACHE_TTL_DAYS`` and ``FAQCACHE_OLLAMA_MODEL``
    are read from the environment; explicit keyword arguments win over both.
    """

    env_map: dict[str, object] = {}
    if os.environ.get("FAQCACHE_THRESHOLD"):
        env_map["threshold"] = float(os.environ["FAQCACHE_THRESHOLD"])
    if os.environ.get("FAQCACHE_TTL_DAYS"):
        env_map["ttl_days"] = int(os.environ["FAQCACHE_TTL_DAYS"])
    if os.environ.get("FAQCACHE_OLLAMA_MODEL"):
        env_map["ollama_model"] = os.environ["FAQCACHE_OLLAMA_MODEL"]
    env_map.update(overrides)
    return Settings(**env_map)  # type: ignore[arg-type]
