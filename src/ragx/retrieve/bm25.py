"""Small deterministic BM25 implementation for the offline baseline."""

import re
from collections import Counter
from dataclasses import dataclass
from math import log

from ragx.chunk.fixed import Chunk

_TERM = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float


class BM25Retriever:
    """BM25 over chunks with stable tie ordering and evidence floors."""

    def __init__(self, chunks: list[Chunk], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self._documents = [Counter(_tokens(chunk.text)) for chunk in chunks]
        self._lengths = [sum(document.values()) for document in self._documents]
        self._average_length = sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
        self._document_frequency = Counter(
            term for document in self._documents for term in document
        )

    def search(
        self, query: str, *, top_k: int = 5, score_floor: float = 0.0
    ) -> list[RetrievedChunk]:
        if not query.strip() or top_k <= 0:
            return []
        query_terms = _tokens(query)
        if not query_terms:
            return []
        scored = [
            RetrievedChunk(chunk, self._score(document, length, query_terms))
            for chunk, document, length in zip(
                self.chunks, self._documents, self._lengths, strict=True
            )
        ]
        return [
            item
            for item in sorted(scored, key=lambda item: (-item.score, item.chunk.id))
            if item.score >= score_floor and item.score > 0
        ][:top_k]

    def _score(self, document: Counter[str], length: int, query_terms: list[str]) -> float:
        if not self.chunks or not self._average_length:
            return 0.0
        score = 0.0
        for term in set(query_terms):
            frequency = document[term]
            if not frequency:
                continue
            inverse_frequency = log(
                1
                + (len(self.chunks) - self._document_frequency[term] + 0.5)
                / (self._document_frequency[term] + 0.5)
            )
            denominator = frequency + self.k1 * (
                1 - self.b + self.b * length / self._average_length
            )
            score += inverse_frequency * frequency * (self.k1 + 1) / denominator
        return score


def _tokens(value: str) -> list[str]:
    return _TERM.findall(value.casefold())
