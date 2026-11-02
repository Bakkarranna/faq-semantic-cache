"""The negation guard, which is what stops opposite questions colliding."""

from __future__ import annotations

import pytest

from faqcache.config import Settings
from faqcache.negation import find_negations, has_negation, looks_like_user_data

WORDS = Settings().negation_words
PATTERNS = Settings().user_data_patterns


@pytest.mark.parametrize(
    "question",
    [
        "refund nahi milega",
        "mujhe refund nahi chahiye",
        "mera parcel kabhi nahi aaya",
        "I do not want to return this",
        "mat bhejo mujhe offers",
        "ORDER nako",
    ],
)
def test_negation_is_detected_in_english_and_hinglish(question: str) -> None:
    assert has_negation(question, WORDS)
    assert find_negations(question, WORDS)


@pytest.mark.parametrize(
    "question",
    [
        "refund kab milega",
        "how long does delivery take",
        "mera order kahan hai",
        "can I pay cash on delivery",
        "size change karwana hai",
    ],
)
def test_plain_questions_are_not_flagged(question: str) -> None:
    assert not has_negation(question, WORDS)
    assert find_negations(question, WORDS) == []


def test_negation_is_matched_as_a_whole_word_only() -> None:
    """A word that merely contains a negation must not be flagged."""

    assert not has_negation("notebook stand chahiye", ("not",))
    assert not has_negation("napkin order status", ("na",))
    assert not has_negation("payment gateway failed", ("mat",))


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Your order ORD392104 shipped on Monday", True),
        ("customer account number 48201933 is locked", True),
        ("the one-time password is 482913", True),
        ("write to bakkarranna at help at example dot in", False),
        ("Refunds are issued within 5 working days", False),
        ("call me on 9876543210 for the delivery", True),
        ("reach us at support@example.com", True),
    ],
)
def test_user_specific_identifiers_are_recognised(text: str, expected: bool) -> None:
    assert looks_like_user_data(text, PATTERNS) is expected


def test_the_guard_catches_the_refund_pair_from_the_brief() -> None:
    positive = "refund milega kya"
    negative = "refund nahi milega kya"
    assert not has_negation(positive, WORDS)
    assert has_negation(negative, WORDS)
