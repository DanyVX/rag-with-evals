# rag-with-evals

A modular Retrieval-Augmented Generation system built as a **measurement instrument**, not a chat demo.

The repository implements ingestion, measurable chunking, dense/BM25/hybrid retrieval, reranking, provider abstraction, citation-aware generation, abstention, prompt-injection fixtures, evaluation metrics, confidence intervals, resumable controlled experiments, reporting utilities, CLI commands, and a thin FastAPI endpoint.

## Architecture

See `docs/ARCHITECTURE.md`.

Core flow:

```text
documents
  -> load / clean / dedupe / manifest
  -> chunk
  -> embed + BM25
  -> FAISS/SQLite + sparse index
  -> dense / BM25 / RRF / MMR / rerank
  -> budgeted prompt with numbered sources
  -> provider adapter
  -> citation validation / abstention
  -> retrieval + generation + cost/latency evaluation
  -> bootstrap CIs / paired tests / reports
```

No end-to-end LangChain or LlamaIndex orchestration is used.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
ragx doctor
```

On Windows:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
ragx doctor
```

## Corpus

The primary pinned corpus is the Python 3.11.17 HTML documentation archive. The PDF/layout stress corpus is NIST AI RMF 1.0.

```bash
python scripts/download_corpus.py
```

The downloader writes SHA-256 digests to `corpus/checksums.json`.

## Basic pipeline

```bash
ragx ingest data/raw/python-3.11.17-docs-html --output data/processed/chunks.jsonl
ragx index data/processed/chunks.jsonl --index-dir data/index
ragx ask "What does Python's context manager protocol do?"
```

## Evaluation

Dataset schema and methodology are documented in `eval/DATASHEET.md`.

Retrieval metrics:
- Recall@1/3/5/10/20
- Hit@k
- MRR
- nDCG@k

Generation/system evaluation includes:
- answer judging hooks
- citation precision/recall primitives
- abstention precision/recall
- prompt-injection attack success rate
- latency
- token/cost accounting structure
- bootstrap confidence intervals
- paired permutation tests
- judge-vs-human Cohen's kappa

Controlled experiment dimensions are in `eval/experiments/grid.json`.

## API

```bash
uvicorn ragx.api:app --reload
```

`POST /ask` returns the answer, citation validity, abstention state, retrieved chunks, scores, and timing fields. The thin API intentionally exposes a retrieval dependency seam so deployments can configure the persisted index/provider without coupling the API to one environment.

## Providers and cost safety

Provider credentials are read from environment variables only.

Real provider calls are disabled while:

```text
RAGX_MAX_SPEND_USD=0
```

Set a positive explicit cap before enabling remote calls.

Adapters included:
- Anthropic API
- OpenAI-compatible endpoint, including local compatible gateways such as Ollama deployments

## Tests / CI

```bash
ruff check .
pytest
```

CI uses deterministic mocked/local behavior. Real-LLM benchmark runs are manual by design.

## Project status

The **software implementation is substantially complete** for v0.1.0.

The repository intentionally does **not** claim empirical Definition-of-Done yet because the project specification requires evidence that cannot be fabricated:

1. at least 100 genuinely human-verified gold questions;
2. a re-review of 30 of them at least one week later;
3. real benchmark runs over the pinned corpus;
4. judge-vs-human agreement;
5. published confidence intervals and injection-test results.

See `docs/RELEASE_CHECKLIST.md` for the exact remaining empirical gates.

## Documentation

- `docs/ARCHITECTURE.md`
- `docs/DECISIONS.md`
- `docs/LIMITATIONS.md`
- `docs/FAILURE_ANALYSIS.md`
- `docs/RELEASE_CHECKLIST.md`
- `eval/DATASHEET.md`
- `results/README.md`
