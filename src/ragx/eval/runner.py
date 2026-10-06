"""Offline evaluation runner for versioned JSONL question sets."""

from __future__ import annotations

import json
from pathlib import Path

from ragx.eval.metrics import hit_at_k, ndcg_at_k, recall_at_k, reciprocal_rank
from ragx.eval.stats import bootstrap_mean_ci
from ragx.pipeline import load_index


def evaluate(index_path: Path, dataset_path: Path, *, k: int = 5) -> dict[str, object]:
    """Evaluate retrieval against JSONL rows with question and gold_chunk_ids fields."""
    retriever = load_index(index_path)
    values: dict[str, list[float]] = {"recall": [], "hit": [], "mrr": [], "ndcg": []}
    count = 0
    for line_number, line in enumerate(
        dataset_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            question = str(row["question"])
            gold = {str(value) for value in row["gold_chunk_ids"]}
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise ValueError(f"Invalid evaluation row {line_number}: {error}") from error
        retrieved = [hit.chunk.id for hit in retriever.search(question, top_k=k)]
        values["recall"].append(recall_at_k(retrieved, gold, k))
        values["hit"].append(hit_at_k(retrieved, gold, k))
        values["mrr"].append(reciprocal_rank(retrieved, gold))
        values["ndcg"].append(ndcg_at_k(retrieved, gold, k))
        count += 1
    if not count:
        raise ValueError("Evaluation dataset contains no records")
    return {
        "n": count,
        "k": k,
        **{
            f"{name}_at_{k}": {"mean": mean, "ci95": [lower, upper]}
            for name, sample in values.items()
            for mean, lower, upper in [bootstrap_mean_ci(sample)]
        },
    }
