"""Sentence embeddings.

Vector texts are L2-normalised so that a dot product equals cosine
similarity, which is what the FAISS index is built on.

A deterministic hashing embedder is used when
``FAQCACHE_EMBEDDER=deterministic``. It maps a text to a stable pseudo-random
vector derived from its tokens. Vectors from different texts are then close to
orthogonal, which makes the unit tests deterministic without downloading a
model.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np

# Kept in sync with README.md. The multilingual model is used because the test
# set contains Hinglish questions, which all-MiniLM-L6-v2 does not embed well.
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

_VECTOR_DIM = 384


class DeterministicEmbedder:
    """A hashing embedder with no model download, for tests only."""

    dim = _VECTOR_DIM

    def encode(self, texts: list[str]) -> np.ndarray:
        rows = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in text.lower().split():
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                bucket = int.from_bytes(digest[:4], "big") % self.dim
                sign = 1.0 if digest[4] & 1 else -1.0
                rows[row, bucket] += sign
            rows[row] += _bias_vector(text)
        return _normalise(rows)


class SentenceTransformerEmbedder:
    """The real model, loaded lazily so imports stay cheap."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        size = getattr(self._model, "get_embedding_dimension", None)
        if size is None:
            size = self._model.get_sentence_embedding_dimension
        self.dim = int(size())

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        vectors = self._model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)


def get_embedder(model_name: str = DEFAULT_MODEL):
    """Return the embedder selected by the environment.

    The switch is read once, so a single test run can use the deterministic
    embedder while the evaluation always uses the real one.
    """

    if os.environ.get("FAQCACHE_EMBEDDER") == "deterministic":
        return DeterministicEmbedder()
    return SentenceTransformerEmbedder(model_name)


def _bias_vector(text: str) -> np.ndarray:
    """A small per-text offset so identical token counts differ."""

    digest = hashlib.sha256(text.encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    return rng.normal(0.0, 0.01, size=_VECTOR_DIM).astype(np.float32)


def _normalise(rows: np.ndarray) -> np.ndarray:
    """Scale every row to unit length, leaving zero rows untouched."""

    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (rows / norms).astype(np.float32)
