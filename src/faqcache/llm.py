"""The LLM client.

Calls a local Ollama server so the measurements in the README were produced
without an API key or a paid account. The client records how many token
estimate it returns so the report can compare LLM calls before and after
caching.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import requests

SYSTEM_PROMPT = (
    "You answer customer support questions for an Indian e-commerce store. "
    "Answer in one or two short plain sentences. Never invent order details, "
    "prices or timings. If the question is about a specific order or account, "
    "say that the customer should open the Orders page to see it."
)


@dataclass(frozen=True)
class Completion:
    """One LLM answer with its cost in tokens."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    model: str


class OllamaClient:
    """A minimal Ollama /api/generate client."""

    def __init__(
        self,
        model: str = "qwen2.5:1.5b",
        url: str = "http://127.0.0.1:11434",
        timeout_s: float = 90.0,
    ) -> None:
        self.model = model
        self.url = url.rstrip("/")
        self.timeout_s = timeout_s

    def generate(self, question: str, context: str | None = None) -> Completion:
        """Answer one question.

        ``context`` is the answer of the matched FAQ when a prototype wants the
        model to reword a cached answer; plain customer questions leave it out.
        """

        prompt = question if context is None else f"{question}\n\nReference: {context}"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": SYSTEM_PROMPT,
            "stream": False,
            "options": {"temperature": 0, "seed": 42},
        }
        response = requests.post(
            f"{self.url}/api/generate",
            json=payload,
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        body = response.json()
        return Completion(
            text=str(body.get("response", "")).strip(),
            prompt_tokens=int(body.get("prompt_eval_count", 0)),
            completion_tokens=int(body.get("eval_count", 0)),
            latency_ms=float(body.get("total_duration", 0)) / 1e6,
            model=self.model,
        )

    def available(self) -> bool:
        """Report whether the local server answers, so callers can skip it."""

        try:
            response = requests.get(f"{self.url}/api/tags", timeout=5.0)
            response.raise_for_status()
        except requests.RequestException:
            return False
        return True


def default_client(settings) -> OllamaClient | None:
    """Return a live client, or None when Ollama is not running."""

    if os.environ.get("FAQCACHE_NO_LLM") == "1":
        return None
    client = OllamaClient(model=settings.ollama_model, url=settings.ollama_url)
    return client if client.available() else None
