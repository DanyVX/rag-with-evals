"""Command-line entry point for ragx."""

from pathlib import Path

import typer

from ragx.eval.experiments import ExperimentConfig, run_experiment
from ragx.eval.report import write_result
from ragx.eval.runner import evaluate
from ragx.eval.stats import bootstrap_mean_ci
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
    """Write a deterministic smoke benchmark artifact for CI plumbing."""
    mean, lower, upper = bootstrap_mean_ci([1.0])
    write_result(
        Path("results/smoke.json"),
        {"smoke_hit_at_1": mean, "ci95": [lower, upper], "n": 1},
        Path.cwd(),
    )
    typer.echo("Wrote results/smoke.json")


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


@app.command(name="eval")
def evaluate_retrieval(
    dataset: Path,
    index_path: Path = Path("cache/index.json"),
    output: Path = Path("results/evaluation.json"),
    k: int = 5,
) -> None:
    """Evaluate a persisted index against a versioned JSONL dataset."""
    metrics = evaluate(index_path, dataset, k=k)
    write_result(output, metrics, Path.cwd())
    typer.echo(f"Wrote {output}")


@app.command()
def report(result: Path = Path("results/evaluation.json")) -> None:
    """Render a saved result artifact without changing any measurement."""
    import json

    payload = json.loads(result.read_text(encoding="utf-8"))
    typer.echo(json.dumps(payload["metrics"], indent=2))


@app.command()
def experiment(dataset: Path, index_path: Path = Path("cache/index.json"), k: int = 5) -> None:
    """Run or resume a hash-addressed retrieval experiment."""
    output, cached = run_experiment(
        ExperimentConfig(str(index_path), str(dataset), k), Path("results")
    )
    typer.echo(f"{'Reused' if cached else 'Wrote'} {output}")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the thin API locally with the offline-safe default configuration."""
    import uvicorn

    uvicorn.run("ragx.api:app", host=host, port=port)
