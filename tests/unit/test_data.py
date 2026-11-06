"""Checks on the two data files the evaluation depends on."""

from __future__ import annotations

import json

import pytest

from faqcache.data import FAQ_PATH, TEST_SET_PATH, label_counts, load_faqs, load_test_set


def test_faq_file_holds_three_hundred_entries() -> None:
    assert len(load_faqs()) == 300


def test_faq_ids_are_unique_and_answers_are_not_empty() -> None:
    faqs = load_faqs()
    assert len({faq.id for faq in faqs}) == len(faqs)
    assert all(faq.answer.strip() for faq in faqs)


def test_every_faq_is_covered_by_at_least_one_test_question() -> None:
    coverage: dict[str, int] = {}
    for item in load_test_set():
        if item.expected == "none":
            continue
        coverage[item.expected] = coverage.get(item.expected, 0) + 1
    assert len(coverage) == 300
    assert min(coverage.values()) >= 1


def test_test_set_holds_one_thousand_questions() -> None:
    assert len(load_test_set()) == 1000


def test_test_set_splits_into_six_hundred_paraphrases_and_four_hundred_negatives() -> None:
    counts = label_counts()
    assert counts == {"total": 1000, "paraphrase": 600, "negative": 400}


def test_every_label_names_a_known_faq_or_none() -> None:
    known = {faq.id for faq in load_faqs()}
    for item in load_test_set():
        assert item.expected == "none" or item.expected in known


def test_the_set_covers_hinglish_questions() -> None:
    languages = {item.language for item in load_test_set()}
    assert "hinglish" in languages
    assert any(item.question.lower().endswith("milega") for item in load_test_set())


def test_a_duplicate_faq_id_is_rejected(tmp_path) -> None:
    broken = tmp_path / "faqs.json"
    broken.write_text(
        json.dumps(
            [
                {"id": "A", "topic": "t", "volatile": False, "question": "q", "answer": "a"},
                {"id": "A", "topic": "t", "volatile": False, "question": "q2", "answer": "a2"},
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate FAQ id"):
        load_faqs(broken)


def test_a_missing_answer_is_rejected(tmp_path) -> None:
    broken = tmp_path / "faqs.json"
    broken.write_text(
        json.dumps([{"id": "A", "topic": "t", "volatile": False, "question": "q?", "answer": " "}]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="empty question or answer"):
        load_faqs(broken)


def test_an_unknown_label_is_rejected(tmp_path) -> None:
    broken = tmp_path / "test_set.json"
    broken.write_text(json.dumps([{"question": "q?", "expected": "FAQ-999"}]), encoding="utf-8")
    with pytest.raises(ValueError, match="not a FAQ id"):
        load_test_set(broken)


def test_a_duplicate_test_question_is_rejected(tmp_path) -> None:
    broken = tmp_path / "test_set.json"
    broken.write_text(
        json.dumps(
            [
                {"question": "same question", "expected": "none"},
                {"question": "same question", "expected": "none"},
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate test question"):
        load_test_set(broken)


def test_the_default_paths_point_at_the_repository() -> None:
    assert FAQ_PATH.name == "faqs.json"
    assert TEST_SET_PATH.name == "test_set.json"
