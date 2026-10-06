from __future__ import annotations

import typer

from ragx.config import get_settings

app = typer.Typer(no_args_is_help=True, help="RAG with rigorous evaluation.")


@app.command()
def doctor() -> None:
    """Validate local configuration without making provider calls."""
    settings = get_settings()
    typer.echo(f"environment={settings.environment}")
    typer.echo(f"max_spend_usd={settings.max_spend_usd:.2f}")
    typer.echo("provider_calls_enabled=" + str(settings.max_spend_usd > 0).lower())


@app.command()
def ingest() -> None:
    """Ingest documents into the canonical document store."""
    typer.echo("ingest: not implemented yet")
    raise typer.Exit(code=2)


@app.command()
def index() -> None:
    """Build sparse and dense indexes."""
    typer.echo("index: not implemented yet")
    raise typer.Exit(code=2)


@app.command()
def ask() -> None:
    """Run a RAG query."""
    typer.echo("ask: not implemented yet")
    raise typer.Exit(code=2)


@app.command()
def eval() -> None:
    """Run evaluation."""
    typer.echo("eval: not implemented yet")
    raise typer.Exit(code=2)


@app.command()
def report() -> None:
    """Build experiment reports."""
    typer.echo("report: not implemented yet")
    raise typer.Exit(code=2)


if __name__ == "__main__":
    app()
