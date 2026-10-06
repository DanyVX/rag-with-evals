from __future__ import annotations

import json
from pathlib import Path

import typer

from ragx.chunk.core import Chunk, ChunkConfig, chunk_document
from ragx.config import get_settings
from ragx.embed.sentence_transformer import SentenceTransformerEmbedder
from ragx.eval.dataset import dataset_checksum, load_jsonl, validate_human_subset
from ragx.index.bm25 import BM25Index
from ragx.index.faiss_sqlite import FaissSQLiteStore
from ragx.ingest.dedupe import deduplicate
from ragx.ingest.loaders import load_document
from ragx.retrieve.core import dense_search, hybrid_rrf, sparse_search

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
    typer.echo(f"environment={settings.environment}")
    typer.echo(f"max_spend_usd={settings.max_spend_usd:.2f}")
    typer.echo("provider_calls_enabled=" + str(settings.max_spend_usd > 0).lower())


@app.command()
def ingest(
    source: Path = typer.Argument(..., exists=True, file_okay=False),
    output: Path = typer.Option(Path("data/processed/chunks.jsonl")),
    strategy: str = typer.Option("fixed"),
    size: int = typer.Option(256),
    overlap: int = typer.Option(32),
) -> None:
    """Load, clean, deduplicate, and chunk a directory of supported documents."""
    documents = []
    for path in sorted(p for p in source.rglob("*") if p.is_file()):
        try:
            documents.extend(load_document(path))
        except ValueError:
            continue
    documents, removed = deduplicate(documents)
    config = ChunkConfig(strategy=strategy, size=size, overlap=overlap)
    chunks = [chunk for doc in documents for chunk in chunk_document(doc, config)]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(chunk.model_dump_json() for chunk in chunks) + ("\n" if chunks else ""),
        encoding="utf-8",
    )
    typer.echo(
        json.dumps(
            {
                "documents": len(documents),
                "deduplicated": len(removed),
                "chunks": len(chunks),
                "output": str(output),
            }
        )
    )


@app.command()
def index(
    chunks_path: Path = typer.Argument(..., exists=True, dir_okay=False),
    index_dir: Path = typer.Option(Path("data/index")),
    model: str = typer.Option("BAAI/bge-small-en-v1.5"),
) -> None:
    """Build a persistent dense index from chunk JSONL."""
    chunks = _load_chunks(chunks_path)
    embedder = SentenceTransformerEmbedder(model)
    vectors = embedder.encode([chunk.text for chunk in chunks])
    store = FaissSQLiteStore(index_dir, model)
    if store.size():
        raise typer.BadParameter("index directory is not empty; use a fresh directory to avoid duplicates")
    store.add(chunks, vectors)
    typer.echo(json.dumps({"indexed": store.size(), "model": model, "index_dir": str(index_dir)}))


@app.command()
def ask(
    question: str = typer.Argument(...),
    chunks_path: Path = typer.Option(Path("data/processed/chunks.jsonl"), exists=True),
    index_dir: Path = typer.Option(Path("data/index"), exists=True),
    model: str = typer.Option("BAAI/bge-small-en-v1.5"),
    top_k: int = typer.Option(5),
) -> None:
    """Retrieve evidence using dense + BM25 hybrid RRF."""
    chunks = _load_chunks(chunks_path)
    embedder = SentenceTransformerEmbedder(model)
    store = FaissSQLiteStore(index_dir, model)
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
