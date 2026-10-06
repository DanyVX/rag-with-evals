"""CPU-only dense-style hashing retrieval, RRF fusion, and MMR diversification."""

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from hashlib import blake2b
from math import sqrt

from ragx.chunk import Chunk
from ragx.retrieve.bm25 import RetrievedChunk

_TERM = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class HashingDenseRetriever:
    """Deterministic fixed-dimensional baseline; not a substitute for a trained embedder."""

    chunks: Sequence[Chunk]
    dimensions: int = 512
    _vectors: tuple[tuple[float, ...], ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.dimensions < 8:
            raise ValueError("dimensions must be at least 8")
        object.__setattr__(
            self, "_vectors", tuple(self._embed(chunk.text) for chunk in self.chunks)
        )

    def _embed(self, text: str) -> tuple[float, ...]:
        vector = [0.0] * self.dimensions
        for token in _TERM.findall(text.casefold()):
            index = (
                int.from_bytes(blake2b(token.encode(), digest_size=8).digest(), "big")
                % self.dimensions
            )
            vector[index] += 1
        norm = sqrt(sum(value * value for value in vector))
        return tuple(value / norm for value in vector) if norm else tuple(vector)

    def search(self, query: str, *, top_k: int = 5) -> list[RetrievedChunk]:
        if not query.strip() or top_k <= 0 or not self.chunks:
            return []
        query_vector = self._embed(query)
        scores = [
            sum(left * right for left, right in zip(vector, query_vector, strict=True))
            for vector in self._vectors
        ]
        results = [
            RetrievedChunk(chunk, float(score))
            for chunk, score in zip(self.chunks, scores, strict=True)
        ]
        return [
            item
            for item in sorted(results, key=lambda item: (-item.score, item.chunk.id))
            if item.score > 0
        ][:top_k]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[RetrievedChunk]], *, constant: int = 60
) -> list[RetrievedChunk]:
    """Fuse rankings by reciprocal rank with deterministic tie-breaking."""
    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item.chunk.id] = scores.get(item.chunk.id, 0.0) + 1 / (constant + rank)
            chunks[item.chunk.id] = item.chunk
    return [
        RetrievedChunk(chunks[key], score)
        for key, score in sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
    ]


def mmr_select(
    query: str, candidates: Sequence[RetrievedChunk], *, top_k: int, diversity: float = 0.5
) -> list[RetrievedChunk]:
    """Lexical MMR selection for diverse evidence; diversity is bounded [0, 1]."""
    if not 0 <= diversity <= 1:
        raise ValueError("diversity must be between 0 and 1")
    selected: list[RetrievedChunk] = []
    remaining = list(candidates)
    while remaining and len(selected) < top_k:
        best = max(
            remaining,
            key=lambda candidate: (
                (1 - diversity) * candidate.score
                - diversity
                * max(
                    (_overlap(candidate.chunk.text, chosen.chunk.text) for chosen in selected),
                    default=0.0,
                ),
                candidate.chunk.id,
            ),
        )
        selected.append(best)
        remaining.remove(best)
    return selected


def _overlap(first: str, second: str) -> float:
    left, right = set(_TERM.findall(first.casefold())), set(_TERM.findall(second.casefold()))
    return len(left & right) / len(left | right) if left and right else 0.0
