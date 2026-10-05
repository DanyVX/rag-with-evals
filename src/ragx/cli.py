"""Command-line entry point for ragx."""

from pathlib import Path

import typer

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
