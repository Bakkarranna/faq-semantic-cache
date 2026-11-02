"""Loading and validating the FAQ set and the labelled test set."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FAQ_PATH = REPO_ROOT / "data" / "faqs.json"
TEST_SET_PATH = REPO_ROOT / "data" / "test_set.json"

# Label used by a test question that must not be served from the cache.
NEGATIVE_LABEL = "none"


@dataclass(frozen=True)
class Faq:
    """One cached question and the answer stored with it."""

    id: str
    topic: str
    volatile: bool
    question: str
    answer: str


@dataclass(frozen=True)
class LabelledQuestion:
    """A test question and the FAQ id it should be served from, or "none"."""

    question: str
    expected: str
    language: str = "english"


def load_faqs(path: Path = FAQ_PATH) -> list[Faq]:
    """Read data/faqs.json and reject anything malformed.

    The uniqueness of the ids and the non-emptiness of the text are the two
    invariants the cache index relies on, so they are checked here once instead
    of at every call site.
    """

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path} must hold a non-empty JSON list")

    faqs: list[Faq] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        for key in ("id", "topic", "volatile", "question", "answer"):
            if key not in item:
                raise ValueError(f"entry {index} is missing the key {key!r}")
        faq_id = str(item["id"])
        if faq_id in seen:
            raise ValueError(f"duplicate FAQ id {faq_id}")
        if not str(item["question"]).strip() or not str(item["answer"]).strip():
            raise ValueError(f"entry {index} has an empty question or answer")
        seen.add(faq_id)
        faqs.append(
            Faq(
                id=faq_id,
                topic=str(item["topic"]),
                volatile=bool(item["volatile"]),
                question=str(item["question"]).strip(),
                answer=str(item["answer"]).strip(),
            )
        )
    return faqs


def load_test_set(path: Path = TEST_SET_PATH) -> list[LabelledQuestion]:
    """Read data/test_set.json.

    Every label must either name an FAQ id present in data/faqs.json or be the
    literal "none", so the evaluation can tell a wrong answer from a question
    that should never have hit the cache.
    """

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path} must hold a non-empty JSON list")

    known = {faq.id for faq in load_faqs()}
    labels: dict[str, LabelledQuestion] = {}
    for index, item in enumerate(raw):
        question = str(item["question"]).strip()
        expected = str(item["expected"]).strip()
        if not question:
            raise ValueError(f"entry {index} has an empty question")
        if expected != NEGATIVE_LABEL and expected not in known:
            raise ValueError(f"entry {index} labels {expected!r}, which is not a FAQ id")
        if question in labels:
            raise ValueError(f"duplicate test question: {question!r}")
        labels[question] = LabelledQuestion(
            question=question,
            expected=expected,
            language=str(item.get("language", "english")),
        )
    return list(labels.values())


def label_counts() -> dict[str, int]:
    """Count paraphrases and negatives in the test set, used by the report."""

    counts = {"total": 0, "paraphrase": 0, "negative": 0}
    for item in load_test_set():
        counts["total"] += 1
        if item.expected == NEGATIVE_LABEL:
            counts["negative"] += 1
        else:
            counts["paraphrase"] += 1
    return counts
