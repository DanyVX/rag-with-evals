from __future__ import annotations

import numpy as np

from ragx.index.base import SearchHit


def apply_score_floor(hits: list[SearchHit], floor: float | None) -> list[SearchHit]:
    if floor is None:
        return hits
    return [h for h in hits if h.score >= floor]


def filter_metadata(
    hits: list[SearchHit],
    *,
    source_contains: str | None = None,
    page: int | None = None,
) -> list[SearchHit]:
    result = hits
    if source_contains:
        needle = source_contains.lower()
        result = [h for h in result if needle in h.chunk.source.lower()]
    if page is not None:
        result = [h for h in result if h.chunk.page == page]
    return result


def mmr_select(
    query_vector: np.ndarray,
    candidate_vectors: np.ndarray,
    hits: list[SearchHit],
    *,
    k: int,
    lambda_mult: float = 0.5,
) -> list[SearchHit]:
    if not hits or k <= 0:
        return []
    if len(hits) != len(candidate_vectors):
        raise ValueError("candidate vector count must match hits")
    q = np.asarray(query_vector, dtype=np.float32).reshape(-1)
    matrix = np.asarray(candidate_vectors, dtype=np.float32)
    qn = np.linalg.norm(q) or 1.0
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = matrix / norms
    q = q / qn
    relevance = matrix @ q

    selected: list[int] = []
    remaining = list(range(len(hits)))
    while remaining and len(selected) < min(k, len(hits)):
        best = None
        best_score = float("-inf")
        for idx in remaining:
            redundancy = max((float(matrix[idx] @ matrix[j]) for j in selected), default=0.0)
            score = lambda_mult * float(relevance[idx]) - (1 - lambda_mult) * redundancy
            if best is None or score > best_score or (
                score == best_score and hits[idx].chunk.id < hits[best].chunk.id
            ):
                best, best_score = idx, score
        assert best is not None
        selected.append(best)
        remaining.remove(best)
    return [
        SearchHit(chunk=hits[i].chunk, score=float(relevance[i]), rank=rank + 1)
        for rank, i in enumerate(selected)
    ]
