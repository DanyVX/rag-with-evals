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

`POST /ask` uses the configured persisted FAISS/SQLite index plus BM25 (dense, sparse, or hybrid RRF), optional cross-encoder reranking, and the configured provider. It returns numbered citations, citation validation, abstention state, retrieved chunks with scores, retrieval/rerank/generation timing, token usage, and per-query cost. `GET /health` reports index/provider/budget state and fails clearly when runtime configuration is incomplete.

## Providers and cost safety

Provider credentials are read from environment variables only.

Real provider calls are disabled while:

```text
RAGX_MAX_SPEND_USD=0
```

Set a positive explicit cap before enabling paid remote calls. Paid providers also require explicit `RAGX_INPUT_USD_PER_MILLION` and `RAGX_OUTPUT_USD_PER_MILLION` values so the runtime can reserve a conservative maximum cost before each request and reconcile it against provider-reported usage. Local OpenAI-compatible endpoints on localhost may run at zero monetary cost.

Key runtime variables include `RAGX_PROVIDER`, `RAGX_MODEL`, `RAGX_CHUNKS_PATH`, `RAGX_INDEX_DIR`, `RAGX_EMBEDDING_MODEL`, `RAGX_RETRIEVER`, `RAGX_TOP_K`, and optional `RAGX_RERANKER_MODEL`. Mock generation is disabled for the API unless `RAGX_ALLOW_MOCK_API=true` is explicitly set.

Adapters included:
- Anthropic API
- OpenAI-compatible endpoint, including local compatible gateways such as Ollama deployments

## Tests / CI

```bash
ruff check .
pytest
```

CI uses deterministic mocked/local behavior. Real-LLM benchmark runs are manual by design.

## Published evaluation results

<!-- RAGX_RESULTS_START -->
Final human-verified results have not been published yet. The release gate will keep v0.1.0 blocked until the required empirical evidence exists.
<!-- RAGX_RESULTS_END -->

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
