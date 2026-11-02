#!/usr/bin/env python
"""Pull real negation-guard examples out of a measured sweep.

Reads the per-threshold request logs a sweep left in results/ and writes the
questions that carried a negation word, with the similarity the model gave the
nearest FAQ and the decision that resulted. Those rows are the evidence for the
negation guard section of the README: a question that carries "not" or "nahi"
and still sits close enough to a stored answer to be a hit without the guard.

    python scripts/pick_negation_examples.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

RESULTS = REPO_ROOT / "results"

FAQS = json.loads((REPO_ROOT / "data" / "faqs.json").read_text(encoding="utf-8"))
BY_ID = {faq["id"]: faq for faq in FAQS}


def main() -> int:
    rows: list[tuple[float, dict]] = []
    for path in sorted(RESULTS.glob("requests_t*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["negations"]:
                rows.append((path.stem, row))

    questions: dict[str, dict] = {}
    for _, row in rows:
        key = row["question"]
        if key in questions:
            continue
        faq = BY_ID.get(row["faq"] or "")
        questions[key] = {
            "question": key,
            "negations": row["negations"],
            "outcome": row["outcome"],
            "similarity": row["similarity"],
            "matched_faq_id": row["faq"],
            "matched_faq": None if faq is None else {
                "id": faq["id"],
                "question": faq["question"],
                "answer": faq["answer"],
                "volatile": faq.get("volatile", False),
            },
            "answer_chars": row["answer_chars"],
            "latency_ms": row["latency_ms"],
        }

    examples = sorted(questions.values(), key=lambda item: -(item["similarity"] or 0.0))
    payload = {
        "negation_questions_measured": len(examples),
        "negated_hits": sum(1 for e in examples if e["outcome"] == "hit"),
        "examples": examples,
    }
    out = RESULTS / "negation_examples.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(examples)} negation-carrying questions, "
          f"{payload['negated_hits']} served from cache, written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
