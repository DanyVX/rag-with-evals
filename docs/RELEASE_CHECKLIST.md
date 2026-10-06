# v0.1.0 release checklist

## Software implementation

- [x] Explicit modular RAG architecture
- [x] MD / HTML / PDF / TXT loading
- [x] cleaning, content hashing, exact and near-duplicate detection
- [x] incremental ingest manifest and deletion detection
- [x] fixed, recursive, sentence-window, and structure-aware chunking
- [x] deterministic chunk IDs
- [x] sentence-transformer embedding interface
- [x] vector-store interface
- [x] persistent FAISS + SQLite implementation
- [x] BM25 sparse retrieval
- [x] dense retrieval
- [x] hybrid Reciprocal Rank Fusion
- [x] MMR utility
- [x] metadata filters and score floors
- [x] cross-encoder reranking
- [x] OpenAI-compatible and Anthropic provider adapters
- [x] environment-only secrets
- [x] spend-cap guardrail
- [x] citation parsing / validation
- [x] explicit abstention phrase
- [x] retrieved-text prompt-injection boundary
- [x] adversarial injection fixtures
- [x] retrieval metrics
- [x] citation / abstention metrics
- [x] bootstrap confidence intervals
- [x] paired permutation test
- [x] judge rubric, parse retry, judge_error behavior, Cohen's kappa
- [x] deterministic experiment config hashing, caching, and resumability
- [x] heatmap, Pareto, and factor-effect report utilities
- [x] thin FastAPI POST /ask
- [x] operational ragx CLI
- [x] deterministic mocked CI path
- [x] architecture, limitations, datasheet, and failure-analysis docs
- [x] pinned corpus download/checksum mechanism

## Empirical acceptance gates

These are deliberately not marked complete until real evidence exists.

- [ ] Download the pinned corpora and commit/archive the generated checksum manifest with the run.
- [ ] Build the synthetic and unanswerable evaluation sets.
- [ ] Manually verify at least 100 gold questions.
- [ ] Re-review 30 of those questions at least one week later and record consistency.
- [ ] Run the complete controlled experiment grid and archive the compact 5,184-config evidence table.
- [ ] Validate the LLM judge against human labels and report agreement / Cohen's kappa.
- [ ] Publish headline metrics with confidence intervals and sample counts.
- [ ] Publish injection attack success rate.
- [ ] Publish failure-rate accounting for provider/judge errors.
- [ ] Tag/release v0.1.0 only after the evidence above is archived.

No checkbox in the empirical section should be completed by inventing data.
