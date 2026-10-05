"""Command-line entry point for ragx."""

from pathlib import Path

import typer

from ragx.ingest.service import ingest_paths
from ragx.pipeline import answer, build_index, load_index
from ragx.provenance import collect_environment
from ragx.settings import Settings

app = typer.Typer(no_args_is_help=True, help="Evaluation-first RAG tooling.")


@app.command()
def doctor() -> None:
    """Validate configuration and print non-sensitive environment provenance."""
    settings = Settings()
    provenance = collect_environment(Path.cwd())
    typer.echo(f"provider={settings.provider}")
    typer.echo(f"provider_spend_cap_usd={settings.max_provider_spend_usd}")
    typer.echo(f"git_revision={provenance['git_revision'] or 'uncommitted-or-not-a-repository'}")
    typer.echo(f"cuda_available={provenance['cuda_available']}")


@app.command()
def benchmark() -> None:
    """Reserve a stable command name for the evaluation harness."""
    typer.echo("Benchmarking is introduced with the evaluation milestone.")
    raise typer.Exit(code=2)


@app.command()
def ingest(path: list[Path]) -> None:
    """Ingest supported local documents and report safety-relevant skips."""
    result = ingest_paths(path)
    typer.echo(f"documents={len(result.documents)} duplicates_skipped={result.duplicates_skipped}")
    for warning in result.warnings:
        typer.echo(f"warning: {warning.source}: {warning.message}")


@app.command()
def index(path: list[Path], output: Path = Path("cache/index.json"), chunk_size: int = 256) -> None:
    """Build a persisted local index from local documents."""
    typer.echo(build_index(path, output, chunk_size=chunk_size))


@app.command()
def ask(query: str, index_path: Path = Path("cache/index.json"), top_k: int = 3) -> None:
    """Ask an evidence-only question of a persisted index."""
    response, hits = answer(load_index(index_path), query, top_k=top_k)
    typer.echo(response)
    for number, hit in enumerate(hits, start=1):
        source = hit.chunk.metadata.get("title", hit.chunk.document_hash)
        typer.echo(f"[{number}] {source} score={hit.score:.3f}")
