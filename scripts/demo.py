#!/usr/bin/env python
"""Walk one conversation through the cache and print what the user gets.

The sequence is chosen to show the three outcomes the cache can return, in the
order a customer meets them:

1. a question the FAQ set does not cover, answered by the LLM and stored;
2. the same question asked in other words, answered from the stored entry;
3. a negated version of an FAQ, which the cache declines to answer at all.

No two of the three questions are word for word the same, so a lexical cache
would answer none of them and the reworded question is what makes step 2 a
semantic hit rather than a dictionary hit.

The model is loaded before the sequence starts, so the timed steps report the
steady state rather than the cost of loading the weights.

    python scripts/demo.py [--json results/demo.json]

Run it with Ollama serving the model in ``settings.ollama_model``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from faqcache.cache import SemanticCache  # noqa: E402
from faqcache.config import load_settings  # noqa: E402
from faqcache.data import load_faqs  # noqa: E402
from faqcache.embeddings import get_embedder  # noqa: E402
from faqcache.llm import OllamaClient, default_client  # noqa: E402

# Measured against the embedder before the run: the cold question scores 0.462
# against its nearest FAQ and the rewording scores 0.850 against the cold one,
# so step 1 is a real miss and step 2 a real hit.
COLD_QUESTION = "do you have a loyalty programme for regular customers"
REWORDING = "what loyalty benefits do regular customers get"
NEGATED = "I do not want to return this, can I exchange instead"


@dataclass(frozen=True)
class Step:
    question: str
    role: str
    outcome: str
    similarity: float | None
    matched_faq_id: str | None
    latency_ms: float
    answer: str
    tokens: int | None


def run_demo(client: OllamaClient | None) -> list[Step]:
    settings = load_settings()
    embedder = get_embedder(settings.model_name)
    cache = SemanticCache(settings, embedder=embedder, client=client)
    cache.warm(load_faqs())
    if client is not None:
        # The answer to the throwaway question is dropped: the first call to a
        # cold model also pays for loading it into memory.
        client.generate("Are you there?")

    steps: list[Step] = []
    plan = (
        ("a question the FAQ set does not cover", COLD_QUESTION),
        ("the same question in other words", REWORDING),
        ("a question carrying a negation word", NEGATED),
    )
    for role, question in plan:
        started = time.perf_counter()
        decision = cache.ask(question)
        answer = decision.answer
        tokens = None
        if decision.completion is not None:
            tokens = decision.completion.prompt_tokens + decision.completion.completion_tokens
        steps.append(
            Step(
                question=question,
                role=role,
                outcome=decision.outcome,
                similarity=decision.similarity,
                matched_faq_id=decision.matched_faq_id,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                answer=answer,
                tokens=tokens,
            )
        )
    if client is not None:
        client.unload()
    return steps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=None, help="write the steps to this file")
    args = parser.parse_args()

    settings = load_settings()
    client = default_client(settings)
    if client is None:
        print("Ollama is not answering on", settings.ollama_url)
        print("a miss then answers with the fallback notice instead of the model.")
    else:
        print(f"model {client.model}, embedder {settings.model_name}")
        print(f"cache entries {len(load_faqs())} FAQs, threshold {settings.threshold}\n")

    steps = run_demo(client)
    for number, step in enumerate(steps, start=1):
        similarity = f"{step.similarity:.3f}" if step.similarity is not None else "   -  "
        tokens = f"{step.tokens} tokens" if step.tokens is not None else "no call"
        print(f"{number}. {step.role}")
        print(f'   "{step.question}"')
        print(f"   {step.outcome}   similarity {similarity}   {step.latency_ms:7.1f} ms   {tokens}")
        if step.matched_faq_id:
            print(f"   served from {step.matched_faq_id}")
        print(f"   {step.answer[:160]}\n")

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "model": settings.ollama_model,
                    "embedder": settings.model_name,
                    "threshold": settings.threshold,
                    "steps": [
                        {
                            "question": step.question,
                            "role": step.role,
                            "outcome": step.outcome,
                            "similarity": step.similarity,
                            "matched_faq_id": step.matched_faq_id,
                            "latency_ms": round(step.latency_ms, 2),
                            "answer_chars": len(step.answer),
                            "tokens": step.tokens,
                        }
                        for step in steps
                    ],
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
