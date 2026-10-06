from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from ragx.chunk.core import ChunkConfig, chunk_document
from ragx.chunk.tokenizer import HuggingFaceTokenCodec
from ragx.embed.cache import CachedEmbedder, EmbeddingCache
from ragx.embed.sentence_transformer import SentenceTransformerEmbedder
from ragx.eval.metrics import bootstrap_ci, paired_permutation_test, retrieval_metrics
from ragx.experiments import ExperimentRunner
from ragx.index.bm25 import BM25Index
from ragx.index.memory import InMemoryVectorStore
from ragx.ingest.models import Document
from ragx.retrieve.core import dense_search, hybrid_rrf, sparse_search
from ragx.retrieve.rerank import CrossEncoderReranker

KS = (1, 3, 5, 10, 20)


def load_documents(path: Path) -> list[Document]:
    return [
        Document.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_items(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def unique_sources(hits) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for hit in hits:
        source = hit.chunk.source
        if source not in seen:
            seen.add(source)
            result.append(source)
    return result


def mean_ci(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"mean": 0.0, "ci95": [0.0, 0.0], "n": 0}
    lo, hi = bootstrap_ci(values, samples=2000, seed=17)
    return {
        "mean": float(np.mean(values)),
        "ci95": [lo, hi],
        "n": len(values),
    }


def canonical_configs(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    base_dimensions = itertools.product(
        grid["chunk_strategy"],
        grid["chunk_size"],
        grid["overlap_fraction"],
        grid["retriever"],
        grid["top_k"],
        grid["embedding_model"],
        grid["metadata_prepend"],
    )
    for (
        strategy,
        size,
        overlap_fraction,
        retriever,
        top_k,
        embedding_model,
        metadata_prepend,
    ) in base_dimensions:
        configs.append(
            {
                "chunk_strategy": strategy,
                "chunk_size": size,
                "overlap_fraction": overlap_fraction,
                "retriever": retriever,
                "reranker": False,
                "rerank_top_n": None,
                "top_k": top_k,
                "embedding_model": embedding_model,
                "metadata_prepend": metadata_prepend,
            }
        )
        for top_n in grid["rerank_top_n"]:
            configs.append(
                {
                    "chunk_strategy": strategy,
                    "chunk_size": size,
                    "overlap_fraction": overlap_fraction,
                    "retriever": retriever,
                    "reranker": True,
                    "rerank_top_n": top_n,
                    "top_k": top_k,
                    "embedding_model": embedding_model,
                    "metadata_prepend": metadata_prepend,
                }
            )
    return configs


class CorpusCache:
    def __init__(
        self,
        documents: list[Document],
        embedding_cache: EmbeddingCache,
    ) -> None:
        self.documents = documents
        self.embedding_cache = embedding_cache
        self._corpora: dict[tuple, tuple] = {}
        self._codecs: dict[str, HuggingFaceTokenCodec] = {}
        self._reranker: CrossEncoderReranker | None = None

    def get(self, config: dict[str, Any]):
        key = (
            config["chunk_strategy"],
            config["chunk_size"],
            config["overlap_fraction"],
            config["embedding_model"],
            config["metadata_prepend"],
        )
        if key in self._corpora:
            return self._corpora[key]

        model_id = config["embedding_model"]
        codec = self._codecs.setdefault(model_id, HuggingFaceTokenCodec(model_id))
        size = int(config["chunk_size"])
        overlap = int(round(size * float(config["overlap_fraction"])))
        chunk_config = ChunkConfig(
            strategy=config["chunk_strategy"],
            size=size,
            overlap=overlap,
            prepend_section=bool(config["metadata_prepend"]),
        )
        chunks = [
            chunk
            for document in self.documents
            for chunk in chunk_document(document, chunk_config, codec=codec)
        ]
        base = SentenceTransformerEmbedder(model_id)
        embedder = CachedEmbedder(
            base,
            self.embedding_cache,
            preprocess_version="experiments-v1",
        )
        vectors = embedder.encode([chunk.text for chunk in chunks])
        store = InMemoryVectorStore(model_id)
        store.add(chunks, vectors)
        bm25 = BM25Index(chunks)
        value = (chunks, embedder, store, bm25)
        self._corpora[key] = value
        return value

    def reranker(self, model_id: str) -> CrossEncoderReranker:
        if self._reranker is None or self._reranker.model_id != model_id:
            self._reranker = CrossEncoderReranker(model_id)
        return self._reranker


def evaluate_config(
    config: dict[str, Any],
    *,
    cache: CorpusCache,
    items: list[dict[str, Any]],
    reranker_model: str,
) -> dict[str, Any]:
    _, embedder, store, bm25 = cache.get(config)
    top_k = int(config["top_k"])
    candidate_k = int(config["rerank_top_n"] or max(top_k, 20))

    per_question: list[dict[str, Any]] = []
    failures = 0
    started_all = perf_counter()
    for item in items:
        if not item.get("answerable", True):
            continue
        gold_sources = set(item.get("gold_sources") or [])
        if not gold_sources:
            continue

        started = perf_counter()
        try:
            if config["retriever"] == "dense":
                candidates = dense_search(
                    item["question"],
                    embedder=embedder,
                    store=store,
                    k=candidate_k,
                )
            elif config["retriever"] == "bm25":
                candidates = sparse_search(item["question"], index=bm25, k=candidate_k)
            else:
                dense = dense_search(
                    item["question"],
                    embedder=embedder,
                    store=store,
                    k=candidate_k,
                )
                sparse = sparse_search(item["question"], index=bm25, k=candidate_k)
                candidates = hybrid_rrf(dense, sparse, k=candidate_k)

            if config["reranker"]:
                hits = cache.reranker(reranker_model).rerank(
                    item["question"],
                    candidates,
                    top_k,
                )
            else:
                hits = candidates[:top_k]

            ranked_sources = unique_sources(hits)
            metric = retrieval_metrics(ranked_sources, gold_sources, ks=KS)
            row = {
                "id": item["id"],
                "mrr": metric.mrr,
                "latency_ms": (perf_counter() - started) * 1000.0,
            }
            row.update({f"recall@{k}": metric.recall_at_k[k] for k in KS})
            row.update({f"hit@{k}": metric.hit_at_k[k] for k in KS})
            row.update({f"ndcg@{k}": metric.ndcg_at_k[k] for k in KS})
            per_question.append(row)
        except Exception as exc:
            failures += 1
            per_question.append(
                {
                    "id": item["id"],
                    "error": repr(exc),
                    "mrr": 0.0,
                    "latency_ms": (perf_counter() - started) * 1000.0,
                    **{f"recall@{k}": 0.0 for k in KS},
                    **{f"hit@{k}": 0.0 for k in KS},
                    **{f"ndcg@{k}": 0.0 for k in KS},
                }
            )

    summary: dict[str, Any] = {
        "question_count": len(per_question),
        "failure_count": failures,
        "failure_rate": failures / len(per_question) if per_question else 0.0,
        "wall_time_s": perf_counter() - started_all,
        "latency_ms": mean_ci([row["latency_ms"] for row in per_question]),
        "mrr": mean_ci([row["mrr"] for row in per_question]),
        "per_question": per_question,
    }
    for k in KS:
        summary[f"recall@{k}"] = mean_ci([row[f"recall@{k}"] for row in per_question])
        summary[f"hit@{k}"] = mean_ci([row[f"hit@{k}"] for row in per_question])
        summary[f"ndcg@{k}"] = mean_ci([row[f"ndcg@{k}"] for row in per_question])
    return summary


def ci_overlap(a: list[float], b: list[float]) -> bool:
    return max(a[0], b[0]) <= min(a[1], b[1])


def build_summary(result_files: list[Path], output_dir: Path) -> dict[str, Any]:
    rows = []
    for path in result_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "ok":
            continue
        rows.append(
            {
                "config_hash": path.stem,
                "config": payload["config"],
                "result": payload["result"],
            }
        )
    if not rows:
        raise ValueError("no successful experiment results")

    best = max(rows, key=lambda row: row["result"]["mrr"]["mean"])
    baseline_candidates = [
        row
        for row in rows
        if row["config"]["chunk_strategy"] == "fixed"
        and row["config"]["chunk_size"] == 256
        and float(row["config"]["overlap_fraction"]) == 0.1
        and row["config"]["retriever"] == "hybrid"
        and row["config"]["reranker"] is False
        and row["config"]["top_k"] == 5
        and row["config"]["metadata_prepend"] is False
    ]
    baseline = baseline_candidates[0] if baseline_candidates else rows[0]

    best_scores = {
        row["id"]: row["mrr"] for row in best["result"]["per_question"]
    }
    baseline_scores = {
        row["id"]: row["mrr"] for row in baseline["result"]["per_question"]
    }
    common = sorted(set(best_scores) & set(baseline_scores))
    p_value = (
        paired_permutation_test(
            [best_scores[item] for item in common],
            [baseline_scores[item] for item in common],
            samples=5000,
            seed=17,
        )
        if common
        else None
    )
    overlap = ci_overlap(best["result"]["mrr"]["ci95"], baseline["result"]["mrr"]["ci95"])

    factors = [
        "chunk_strategy",
        "chunk_size",
        "overlap_fraction",
        "retriever",
        "reranker",
        "rerank_top_n",
        "top_k",
        "embedding_model",
        "metadata_prepend",
    ]
    effects = []
    for factor in factors:
        groups: dict[str, list[float]] = {}
        for row in rows:
            key = str(row["config"].get(factor))
            groups.setdefault(key, []).append(float(row["result"]["mrr"]["mean"]))
        means = {key: float(np.mean(values)) for key, values in groups.items()}
        effect = max(means.values()) - min(means.values()) if len(means) > 1 else 0.0
        effects.append({"factor": factor, "effect_range_mrr": effect, "means": means})
    effects.sort(key=lambda item: item["effect_range_mrr"], reverse=True)

    summary = {
        "configs_completed": len(rows),
        "best": best,
        "baseline": baseline,
        "best_vs_baseline": {
            "paired_permutation_p_value": p_value,
            "mrr_ci_overlap": overlap,
            "interpretation": (
                "difference not significant: 95% confidence intervals overlap"
                if overlap
                else "confidence intervals do not overlap; inspect paired p-value"
            ),
            "paired_question_count": len(common),
        },
        "factor_effects_ranked": effects,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    sizes = sorted({int(row["config"]["chunk_size"]) for row in rows})
    overlaps = sorted({float(row["config"]["overlap_fraction"]) for row in rows})
    matrix = np.full((len(overlaps), len(sizes)), np.nan)
    for oi, overlap in enumerate(overlaps):
        for si, size in enumerate(sizes):
            values = [
                row["result"]["mrr"]["mean"]
                for row in rows
                if int(row["config"]["chunk_size"]) == size
                and float(row["config"]["overlap_fraction"]) == overlap
            ]
            if values:
                matrix[oi, si] = float(np.mean(values))
    fig, ax = plt.subplots()
    image = ax.imshow(matrix, aspect="auto")
    ax.set_xticks(range(len(sizes)), labels=[str(size) for size in sizes])
    ax.set_yticks(range(len(overlaps)), labels=[f"{overlap:.0%}" for overlap in overlaps])
    ax.set_xlabel("Chunk size (tokens)")
    ax.set_ylabel("Overlap")
    ax.set_title("Mean MRR by chunk size and overlap")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / "chunk_heatmap.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.scatter(
        [row["result"]["latency_ms"]["mean"] for row in rows],
        [row["result"]["mrr"]["mean"] for row in rows],
    )
    ax.set_xlabel("Mean retrieval latency (ms)")
    ax.set_ylabel("MRR")
    ax.set_title("Retrieval quality vs latency")
    fig.tight_layout()
    fig.savefig(output_dir / "pareto_quality_latency.png", dpi=160)
    plt.close(fig)

    lines = [
        "# Controlled retrieval experiment summary",
        "",
        f"Completed configurations: {len(rows)}",
        "",
        f"Best MRR: {best['result']['mrr']['mean']:.4f} "
        f"(95% CI {best['result']['mrr']['ci95'][0]:.4f}–"
        f"{best['result']['mrr']['ci95'][1]:.4f})",
        "",
        f"Baseline MRR: {baseline['result']['mrr']['mean']:.4f} "
        f"(95% CI {baseline['result']['mrr']['ci95'][0]:.4f}–"
        f"{baseline['result']['mrr']['ci95'][1]:.4f})",
        "",
        "Best vs baseline: "
        + summary["best_vs_baseline"]["interpretation"]
        + (
            f"; paired permutation p={p_value:.6f}."
            if p_value is not None
            else "."
        ),
        "",
        "## What actually mattered",
        "",
        "| Factor | MRR effect range |",
        "|---|---:|",
    ]
    for item in effects:
        lines.append(f"| {item['factor']} | {item['effect_range_mrr']:.4f} |")
    (output_dir / "EXPERIMENT_SUMMARY.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("documents", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--grid", type=Path, default=Path("eval/experiments/grid.json"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/experiments/runs"))
    parser.add_argument("--summary-dir", type=Path, default=Path("results/experiments"))
    parser.add_argument("--embedding-cache", type=Path, default=Path("cache/experiment_embeddings.sqlite3"))
    parser.add_argument("--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--strategy")
    parser.add_argument("--size", type=int)
    parser.add_argument("--embedding-model")
    parser.add_argument("--max-configs", type=int, default=0)
    args = parser.parse_args()

    documents = load_documents(args.documents)
    items = load_items(args.dataset)
    grid = json.loads(args.grid.read_text(encoding="utf-8"))
    configs = canonical_configs(grid)
    if args.strategy:
        configs = [c for c in configs if c["chunk_strategy"] == args.strategy]
    if args.size:
        configs = [c for c in configs if int(c["chunk_size"]) == args.size]
    if args.embedding_model:
        configs = [c for c in configs if c["embedding_model"] == args.embedding_model]
    expected = len(configs)
    if args.max_configs > 0:
        configs = configs[: args.max_configs]

    cache = CorpusCache(documents, EmbeddingCache(args.embedding_cache))
    runner = ExperimentRunner(args.results_dir)
    for index, config in enumerate(configs, 1):
        print(f"[{index}/{len(configs)}] {json.dumps(config, sort_keys=True)}")
        runner.run(
            config,
            lambda active, cache=cache: evaluate_config(
                active,
                cache=cache,
                items=items,
                reranker_model=args.reranker_model,
            ),
        )

    result_files = sorted(args.results_dir.glob("*.json"))
    summary = build_summary(result_files, args.summary_dir)
    summary["configs_expected_for_invocation"] = expected
    summary["configs_requested"] = len(configs)
    (args.summary_dir / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "configs_expected_for_invocation": expected,
        "configs_requested": len(configs),
        "configs_completed_total": summary["configs_completed"],
        "summary_dir": str(args.summary_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
