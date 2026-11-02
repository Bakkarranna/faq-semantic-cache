"""The negation guard.

``"refund milega"`` and ``"refund nahi milega"`` are opposite questions, and
cosine similarity cannot tell them apart. Any question carrying a negation
word is therefore refused by the cache, whatever the similarity says, and is
sent to the LLM instead. That trades a small hit rate for correctness on
questions where a cached answer would state the opposite of what the customer
asked.
"""

from __future__ import annotations

import re


def has_negation(text: str, negation_words: tuple[str, ...]) -> bool:
    """Report whether any negation word appears as a whole word."""

    lowered = text.lower()
    # Hinglish negation sometimes attaches to the verb, so the words are also
    # matched as prefixes of longer romanised words such as "nahi" in "nahin".
    for word in negation_words:
        if re.search(rf"(?<![a-z]){re.escape(word)}(?![a-z])", lowered):
            return True
        if len(word) > 3 and re.search(rf"(?<![a-z]){re.escape(word)}[a-z]*", lowered):
            return True
    return False


def find_negations(text: str, negation_words: tuple[str, ...]) -> list[str]:
    """Return the negation words found in a text, for the log record."""

    found: list[str] = []
    lowered = text.lower()
    for word in negation_words:
        if re.search(rf"(?<![a-z]){re.escape(word)}(?![a-z])", lowered):
            found.append(word)
        elif len(word) > 3 and re.search(rf"(?<![a-z]){re.escape(word)}[a-z]*", lowered):
            found.append(f"{word}*")
    return found


def looks_like_user_data(text: str, patterns: tuple[str, ...]) -> bool:
    """Report whether a text contains customer-specific identifiers."""

    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)
