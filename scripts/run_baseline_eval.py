from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ragx.chunk.core import Chunk
from ragx.embed.sentence_transformer import SentenceTransformerEmbedder
from ragx.eval.metrics import bootstrap_ci, retrieval_metrics
from ragx.index.bm25 import BM25Index
from ragx.index.memory import InMemoryVectorStore
from ragx.retrieve.core import dense_search, hybrid_rrf, sparse_search

KS = (1, 3, 5, 10, 20)


def load_chunks(path: Path) -> list[Chunk]:
    return [
        Chunk.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_items(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def summarize(rows: list[dict]) -> dict:
    keys = ["mrr"] + [f"recall@{k}" for k in KS] + [f"ndcg@{k}" for k in KS]
    out = {"n": len(rows)}
    for key in keys:
        values = [float(row[key]) for row in rows]
        mean = float(np.mean(values)) if values else 0.0
        lo, hi = bootstrap_ci(values, samples=2000, seed=17) if values else (0.0, 0.0)
        out[key] = {"mean": mean, "ci95": [lo, hi]}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("chunks", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/bootstrap/retrieval_baseline.json"))
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    args = parser.parse_args()

    chunks = load_chunks(args.chunks)
    items = [x for x in load_items(args.dataset) if x["answerable"]]
    bm25 = BM25Index(chunks)

    embedder = SentenceTransformerEmbedder(args.model)
    vectors = embedder.encode([c.text for c in chunks])
    store = InMemoryVectorStore(args.model)
    store.add(chunks, vectors)

    all_rows = {"bm25": [], "dense": [], "hybrid": []}
    for item in items:
        question = item["question"]
        gold = set(item["gold_chunk_ids"])
        sparse = sparse_search(question, index=bm25, k=20)
        dense = dense_search(question, embedder=embedder, store=store, k=20)
        hybrid = hybrid_rrf(dense, sparse, k=20)
        for name, hits in [("bm25", sparse), ("dense", dense), ("hybrid", hybrid)]:
            metric = retrieval_metrics([h.chunk.id for h in hits], gold, ks=KS)
            row = {"id": item["id"], "mrr": metric.mrr}
            row.update({f"recall@{k}": metric.recall_at_k[k] for k in KS})
            row.update({f"ndcg@{k}": metric.ndcg_at_k[k] for k in KS})
            all_rows[name].append(row)

    result = {
        "embedding_model": args.model,
        "question_count": len(items),
        "retrievers": {name: summarize(rows) for name, rows in all_rows.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
