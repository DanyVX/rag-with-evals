# rag-with-evals

An inspectable RAG pipeline whose primary product is trustworthy measurement: retrieval, faithfulness, abstention, cost, latency, and controlled comparisons.

> **Status:** M0 foundation. No benchmark claims have been measured yet.

## Why this exists

RAG demos often show a single attractive answer. This repository is designed to answer the harder questions: whether the evidence was retrieved, whether an answer is supported by it, when the system should abstain, and whether a configuration change is statistically meaningful.

## Architecture

```mermaid
flowchart LR
  docs[Versioned source documents] --> ingest[Ingest + clean + dedupe]
  ingest --> chunk[Pluggable chunking]
  chunk --> index[Dense / BM25 indexes]
  query[Query] --> retrieve[Dense / sparse / hybrid retrieval]
  index --> retrieve
  retrieve --> generate[Provider-independent generation]
  generate --> answer[Answer + validated citations]
  retrieve --> eval[Evaluation harness]
  answer --> eval
  eval --> results[Versioned results with CIs]
```

## Quickstart

Requires Python 3.11–3.13 and [uv](https://docs.astral.sh/uv/).

```powershell
uv sync --extra dev
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
```

The default configuration is offline and uses no provider credentials. Provider calls remain blocked until a nonzero spend cap and an explicit provider are configured.

## Design decisions

The initial local index is deliberately CPU-first and file-backed; Docker is not available in the discovered environment. See [docs/DESIGN.md](docs/DESIGN.md) for the full decision log and [docs/LIMITATIONS.md](docs/LIMITATIONS.md) for current limits.

## License

MIT. See [LICENSE](LICENSE).

