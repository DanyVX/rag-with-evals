from __future__ import annotations

import json
from pathlib import Path

import typer

from ragx.chunk.core import Chunk, ChunkConfig, chunk_document
from ragx.chunk.tokenizer import HuggingFaceTokenCodec
from ragx.config import get_settings
from ragx.cost import BudgetExceededError
from ragx.embed.cache import CachedEmbedder, EmbeddingCache
from ragx.embed.sentence_transformer import SentenceTransformerEmbedder
from ragx.eval.dataset import dataset_checksum, load_jsonl, validate_human_subset
from ragx.index.bm25 import BM25Index
from ragx.index.faiss_sqlite import FaissSQLiteStore
from ragx.ingest.incremental import incremental_ingest_directory
from ragx.retrieve.core import dense_search, hybrid_rrf, sparse_search
from ragx.runtime import RAGRuntime

app = typer.Typer(no_args_is_help=True, help="RAG with rigorous evaluation.")


def _load_chunks(path: Path) -> list[Chunk]:
    return [
        Chunk.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@app.command()
def doctor() -> None:
    """Validate local configuration without making provider calls."""
    settings = get_settings()
    typer.echo(
        json.dumps(
            {
                "environment": settings.environment,
                "provider": settings.provider,
                "model": settings.model,
                "embedding_model": settings.embedding_model,
                "embedding_cache_path": str(settings.embedding_cache_path),
                "embedding_preprocess_version": settings.embedding_preprocess_version,
                "retriever": settings.retriever,
                "top_k": settings.top_k,
                "chunks_path": str(settings.chunks_path),
                "index_dir": str(settings.index_dir),
                "max_spend_usd": settings.max_spend_usd,
                "zero_cost_local_endpoint": settings.zero_cost_local_endpoint,
                "mock_api_allowed": settings.allow_mock_api,
            },
            indent=2,
        )
    )


@app.command()
def ingest(
    source: Path = typer.Argument(..., exists=True, file_okay=False),
    output: Path = typer.Option(Path("data/processed/chunks.jsonl")),
    strategy: str = typer.Option("fixed"),
    size: int = typer.Option(256),
    overlap: int = typer.Option(32),
    tokenizer_model: str = typer.Option("BAAI/bge-small-en-v1.5"),
    manifest: Path | None = typer.Option(None),
    document_store: Path | None = typer.Option(None),
) -> None:
    """Incrementally load, clean, deduplicate, and chunk supported documents."""
    config = ChunkConfig(strategy=strategy, size=size, overlap=overlap)
    codec = HuggingFaceTokenCodec(tokenizer_model)
    manifest_path = manifest or output.with_name(output.stem + ".manifest.json")
    documents_path = document_store or output.with_name(output.stem + ".documents.jsonl")
    result = incremental_ingest_directory(
        source,
        output_chunks=output,
        document_store=documents_path,
        manifest_path=manifest_path,
        config=config,
        codec=codec,
        tokenizer_id=tokenizer_model,
    )
    typer.echo(
        json.dumps(
            {
                "files_seen": result.files_seen,
                "changed_files": result.changed_files,
                "deleted_files": result.deleted_files,
                "documents": result.documents,
                "deduplicated_documents": result.deduplicated_documents,
                "chunks": result.chunks,
                "reused_chunks": result.reused_chunks,
                "rebuilt_chunks": result.rebuilt_chunks,
                "tokenizer_model": tokenizer_model,
                "tokenizer_max_sequence_length": codec.max_sequence_length,
                "output": str(output),
                "document_store": str(documents_path),
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )


@app.command()
def index(
    chunks_path: Path = typer.Argument(..., exists=True, dir_okay=False),
    index_dir: Path = typer.Option(Path("data/index")),
    model: str = typer.Option("BAAI/bge-small-en-v1.5"),
    cache_path: Path = typer.Option(Path("cache/embeddings.sqlite3")),
    preprocess_version: str = typer.Option("v1"),
) -> None:
    """Build a persistent dense index from chunk JSONL."""
    chunks = _load_chunks(chunks_path)
    embedder = CachedEmbedder(
        SentenceTransformerEmbedder(model),
        EmbeddingCache(cache_path),
        preprocess_version=preprocess_version,
    )
    vectors = embedder.encode([chunk.text for chunk in chunks])
    store = FaissSQLiteStore(index_dir, model)
    if store.size():
        raise typer.BadParameter("index directory is not empty; use a fresh directory to avoid duplicates")
    store.add(chunks, vectors)
    typer.echo(json.dumps({"indexed": store.size(), "model": model, "index_dir": str(index_dir)}))


@app.command()
def search(
    question: str = typer.Argument(...),
    chunks_path: Path = typer.Option(Path("data/processed/chunks.jsonl"), exists=True),
    index_dir: Path = typer.Option(Path("data/index"), exists=True),
    model: str = typer.Option("BAAI/bge-small-en-v1.5"),
    top_k: int = typer.Option(5),
    cache_path: Path = typer.Option(Path("cache/embeddings.sqlite3")),
    preprocess_version: str = typer.Option("v1"),
) -> None:
    """Retrieve evidence using dense + BM25 hybrid RRF without generation."""
    chunks = _load_chunks(chunks_path)
    embedder = CachedEmbedder(
        SentenceTransformerEmbedder(model),
        EmbeddingCache(cache_path),
        preprocess_version=preprocess_version,
    )
    store = FaissSQLiteStore(index_dir, model)
    if not store.matches_chunks(chunks):
        raise typer.BadParameter("dense index is stale relative to the chunk file; rebuild it")
    bm25 = BM25Index(chunks)
    dense = dense_search(question, embedder=embedder, store=store, k=max(top_k, 20))
    sparse = sparse_search(question, index=bm25, k=max(top_k, 20))
    hits = hybrid_rrf(dense, sparse, k=top_k)
    typer.echo(
        json.dumps(
            [
                {
                    "rank": hit.rank,
                    "score": hit.score,
                    "chunk_id": hit.chunk.id,
                    "source": hit.chunk.source,
                    "text": hit.chunk.text,
                }
                for hit in hits
            ],
            indent=2,
        )
    )


@app.command()
def ask(question: str = typer.Argument(...)) -> None:
    """Run the configured end-to-end RAG query."""
    try:
        runtime = RAGRuntime(get_settings())
        result = runtime.ask(question)
    except (RuntimeError, ValueError, BudgetExceededError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    typer.echo(
        json.dumps(
            {
                "answer": result.answer,
                "citations": result.citations.cited,
                "invalid_citations": result.citations.invalid,
                "abstained": result.abstained,
                "retrieved": [
                    {
                        "rank": hit.rank,
                        "score": hit.score,
                        "chunk_id": hit.chunk.id,
                        "source": hit.chunk.source,
                    }
                    for hit in result.retrieved
                ],
                "timing_ms": {
                    "preprocess": result.preprocess_ms,
                    "retrieval": result.retrieval_ms,
                    "rerank": result.rerank_ms,
                    "generation": result.generation_ms,
                    "total": result.total_ms,
                },
                "usage": {
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "cost_usd": result.cost_usd,
                },
            },
            indent=2,
        )
    )


@app.command("eval")
def eval_command(dataset: Path = typer.Argument(..., exists=True, dir_okay=False)) -> None:
    """Validate and checksum an evaluation dataset before an experiment run."""
    items = load_jsonl(dataset)
    checksum = dataset_checksum(items)
    human_error = None
    try:
        validate_human_subset(items)
    except ValueError as exc:
        human_error = str(exc)
    typer.echo(
        json.dumps(
            {
                "items": len(items),
                "sha256": checksum,
                "human_subset_valid": human_error is None,
                "human_subset_error": human_error,
            },
            indent=2,
        )
    )


@app.command()
def report(results_dir: Path = typer.Argument(Path("results/runs"))) -> None:
    """Summarize cached experiment result files."""
    files = sorted(results_dir.glob("*.json")) if results_dir.exists() else []
    ok = failed = 0
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") == "ok":
            ok += 1
        else:
            failed += 1
    typer.echo(json.dumps({"runs": len(files), "ok": ok, "failed": failed}, indent=2))


if __name__ == "__main__":
    app()
