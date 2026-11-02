"""Checks on the two data files the evaluation depends on."""

from __future__ import annotations

import json

import pytest

from faqcache.data import FAQ_PATH, TEST_SET_PATH, label_counts, load_faqs, load_test_set


def test_faq_file_holds_thirty_entries() -> None:
    assert len(load_faqs()) == 30


def test_faq_ids_are_unique_and_answers_are_not_empty() -> None:
    faqs = load_faqs()
    assert len({faq.id for faq in faqs}) == len(faqs)
    assert all(faq.answer.strip() for faq in faqs)


def test_test_set_holds_one_hundred_questions() -> None:
    assert len(load_test_set()) == 100


def test_test_set_splits_into_sixty_paraphrases_and_forty_negatives() -> None:
    counts = label_counts()
    assert counts == {"total": 100, "paraphrase": 60, "negative": 40}


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
    broken.write_text(json.dumps([{"question": "q?", "expected": "FAQ-99"}]), encoding="utf-8")
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
