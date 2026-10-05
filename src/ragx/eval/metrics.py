"""Retrieval and abstention metrics with no LLM judge dependency."""

from collections.abc import Sequence


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
