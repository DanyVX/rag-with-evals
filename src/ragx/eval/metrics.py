"""Retrieval and abstention metrics with no LLM judge dependency."""

from collections.abc import Sequence
from math import log2


def recall_at_k(retrieved: Sequence[str], gold: set[str], k: int) -> float:
    if not gold:
        return 1.0
    return len(set(retrieved[:k]) & gold) / len(gold)


def hit_at_k(retrieved: Sequence[str], gold: set[str], k: int) -> float:
    return float(bool(set(retrieved[:k]) & gold))


def reciprocal_rank(retrieved: Sequence[str], gold: set[str]) -> float:
    for position, item in enumerate(retrieved, start=1):
        if item in gold:
            return 1 / position
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], gold: set[str], k: int) -> float:
    """Binary-relevance normalized discounted cumulative gain."""
    if not gold or k <= 0:
        return 0.0
    discounted = sum(
        1 / log2(position + 1)
        for position, item in enumerate(retrieved[:k], start=1)
        if item in gold
    )
    ideal = sum(1 / log2(position + 1) for position in range(1, min(len(gold), k) + 1))
    return discounted / ideal if ideal else 0.0


def citation_scores(citations: Sequence[int], source_count: int) -> dict[str, float]:
    """Score citation syntax validity; semantic support requires a separate judge."""
    if not citations:
        return {"precision": 0.0, "recall": 0.0}
    valid = sum(1 <= citation <= source_count for citation in citations)
    return {
        "precision": valid / len(citations),
        "recall": valid / source_count if source_count else 0.0,
    }


def abstention_scores(
    predicted_abstentions: Sequence[bool], expected_abstentions: Sequence[bool]
) -> dict[str, float]:
    if len(predicted_abstentions) != len(expected_abstentions):
        raise ValueError("predicted and expected abstention labels must have equal lengths")
    true_positive = sum(
        predicted and expected
        for predicted, expected in zip(predicted_abstentions, expected_abstentions, strict=True)
    )
    predicted_positive = sum(predicted_abstentions)
    expected_positive = sum(expected_abstentions)
    return {
        "precision": true_positive / predicted_positive if predicted_positive else 0.0,
        "recall": true_positive / expected_positive if expected_positive else 0.0,
    }
