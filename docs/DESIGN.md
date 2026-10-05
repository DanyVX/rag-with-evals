# Design

## Principles

- Measure first: benchmark claims are generated from versioned result artifacts, never entered manually.
- CPU-first by default: the baseline must run without CUDA or Docker.
- Provider isolation: generation is behind an interface and billable calls require an explicit positive cap.
- Deterministic provenance: benchmark artifacts record OS, Python, library versions, hardware, and Git revision.

## Decision log

### 2026-10-05 — Start with a local, file-backed vector-store implementation

**Options:** Qdrant in Docker; FAISS plus SQLite; a hosted vector database.

**Choice:** implement a local FAISS-plus-SQLite-compatible store behind `VectorStore` first. The current Windows environment has no Docker and no CUDA. A local store makes CI and the tiny fixture pipeline reproducible. A Qdrant adapter can be added without changing retrieval callers.

### 2026-10-05 — Select Python documentation as the primary corpus

**Options:** Python documentation; Wikipedia subset; arXiv abstracts; Linux man pages.

**Choice:** use a pinned Python documentation release after M1's downloader records license and checksums. It provides structured HTML, thousands of meaningful chunks, stable public provenance, and realistic exact-ID/API questions. It is not a general-knowledge benchmark; that limitation will be reported.

### 2026-10-05 — Default provider budget is USD 0

**Options:** a small default allowance; an uncapped provider; no default allowance.

**Choice:** zero. Tests use recorded/mock fixtures. A real run requires both an explicit provider and a positive `RAGX_MAX_PROVIDER_SPEND_USD`; the runtime will enforce the ceiling.

