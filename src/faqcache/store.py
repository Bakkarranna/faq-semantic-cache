"""The vector store.

An in-memory FAISS ``IndexFlatIP`` holds one L2-normalised vector per cached
question, so a nearest-neighbour search returns cosine similarity directly.
The same interface is what a RedisVL ``SemanticCache`` exposes in production,
which is the swap called for in the README.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # pragma: no cover - the failure path is covered by the guard tests
    import faiss
except ImportError:  # pragma: no cover
    faiss = None  # type: ignore[assignment]


@dataclass(frozen=True)
class CacheEntry:
    """One stored question, its answer and its metadata."""

    faq_id: str
    question: str
    answer: str
    volatile: bool
    stored_at: float


@dataclass(frozen=True)
class Match:
    """The result of a nearest-neighbour search."""

    entry: CacheEntry
    similarity: float
    position: int


class VectorStore:
    """A brute-force cosine search over unit vectors.

    The index is exact rather than approximate: with tens of cached questions
    the recall of an approximate index changes nothing, and exactness removes
    one variable from the threshold sweep.
    """

    def __init__(self, dim: int) -> None:
        if faiss is None:
            raise RuntimeError("faiss is not installed; run pip install faiss-cpu")
        self.dim = dim
        self._index = faiss.IndexFlatIP(dim)
        self._entries: list[CacheEntry] = []

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> list[CacheEntry]:
        """A copy of the stored entries in insertion order."""

        return list(self._entries)

    def add(self, vector: np.ndarray, entry: CacheEntry) -> None:
        """Add one unit vector and its entry."""

        self._index.add(np.asarray(vector, dtype="float32").reshape(1, self.dim))
        self._entries.append(entry)

    def remove_at(self, position: int) -> None:
        """Remove the entry at one position and rebuild the index.

        ``IndexFlatIP`` has no remove method, so the remaining vectors are
        re-added. The store holds tens of entries, which keeps this cheap.
        """

        if not 0 <= position < len(self._entries):
            raise IndexError(f"no entry at position {position}")
        vectors = self._index.reconstruct_n(0, len(self._entries))
        kept = np.delete(vectors, position, axis=0)
        self._index.reset()
        if kept.size:
            self._index.add(kept)
        self._entries = [entry for index, entry in enumerate(self._entries) if index != position]

    def search(self, vector: np.ndarray, k: int = 1) -> list[Match]:
        """Return the k nearest entries, best first.

        An empty index returns an empty list, so a cold cache reads as a miss
        instead of an error.
        """

        if not self._entries:
            return []
        k = min(k, len(self._entries))
        query = np.asarray(vector, dtype="float32").reshape(1, self.dim)
        scores, positions = self._index.search(query, k)
        return [
            Match(
                entry=self._entries[int(position)],
                similarity=float(score),
                position=int(position),
            )
            for score, position in zip(scores[0], positions[0], strict=True)
            if position >= 0
        ]

    def reset(self) -> None:
        """Drop every entry, leaving the index usable."""

        self._index.reset()
        self._entries = []
