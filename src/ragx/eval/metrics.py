from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class RetrievalMetrics:
    recall_at_k: dict[int, float]
    hit_at_k: dict[int, float]
    mrr: float
    ndcg_at_k: dict[int, float]


def _dcg(relevances: list[int]) -> float:
    if not relevances:
        return 0.0
    return float(sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances)))


def retrieval_metrics(
    ranked_ids: list[str],
    gold_ids: set[str],
    *,
    ks: tuple[int, ...] = (1, 3, 5, 10, 20),
) -> RetrievalMetrics:
    if not gold_ids:
        raise ValueError("gold_ids must not be empty")
    recall: dict[int, float] = {}
    hit: dict[int, float] = {}
    ndcg: dict[int, float] = {}
    first_rank: int | None = None
    for i, item_id in enumerate(ranked_ids, 1):
        if item_id in gold_ids:
            first_rank = first_rank or i

    for k in ks:
        top = ranked_ids[:k]
        found = sum(1 for x in top if x in gold_ids)
        recall[k] = found / len(gold_ids)
        hit[k] = float(found > 0)
        rels = [1 if x in gold_ids else 0 for x in top]
        ideal = [1] * min(len(gold_ids), k)
        denom = _dcg(ideal)
        ndcg[k] = _dcg(rels) / denom if denom else 0.0

    return RetrievalMetrics(
        recall_at_k=recall,
        hit_at_k=hit,
        mrr=(1.0 / first_rank) if first_rank else 0.0,
        ndcg_at_k=ndcg,
    )


def citation_metrics(cited: Iterable[int], supported: Iterable[int], total_claims: int) -> dict[str, float]:
    cited_set = set(cited)
    supported_set = set(supported)
    tp = len(cited_set & supported_set)
    precision = tp / len(cited_set) if cited_set else 0.0
    recall = tp / total_claims if total_claims else 0.0
    return {"citation_precision": precision, "citation_recall": recall}


def abstention_metrics(
    predictions_abstain: list[bool],
    gold_unanswerable: list[bool],
) -> dict[str, float]:
    if len(predictions_abstain) != len(gold_unanswerable):
        raise ValueError("prediction and gold lengths differ")
    tp = sum(p and g for p, g in zip(predictions_abstain, gold_unanswerable))
    fp = sum(p and not g for p, g in zip(predictions_abstain, gold_unanswerable))
    fn = sum((not p) and g for p, g in zip(predictions_abstain, gold_unanswerable))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"abstention_precision": precision, "abstention_recall": recall}


def bootstrap_ci(
    values: list[float],
    *,
    confidence: float = 0.95,
    samples: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    if not values:
        raise ValueError("values must not be empty")
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    means = np.empty(samples, dtype=float)
    for i in range(samples):
        means[i] = rng.choice(arr, size=len(arr), replace=True).mean()
    alpha = 1.0 - confidence
    return (
        float(np.quantile(means, alpha / 2)),
        float(np.quantile(means, 1 - alpha / 2)),
    )


def paired_permutation_test(
    a: list[float],
    b: list[float],
    *,
    samples: int = 5000,
    seed: int = 0,
) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("paired samples must be non-empty and equal length")
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    observed = abs(float(diff.mean()))
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(samples):
        signs = rng.choice(np.array([-1.0, 1.0]), size=len(diff))
        if abs(float((diff * signs).mean())) >= observed:
            extreme += 1
    return (extreme + 1) / (samples + 1)
