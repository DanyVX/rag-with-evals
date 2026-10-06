# Limitations

- v0.1.0 is a sparse BM25 baseline. Dense, hybrid, MMR, and cross-encoder reranking comparisons are not yet implemented.
- The planned primary corpus is English technical documentation, so results will not establish general-domain or multilingual performance.
- The environment has no detected CUDA GPU or Docker installation. The baseline is CPU-first; accelerator comparisons require a separately documented environment.
- No licensed external corpus or human-verified 100-question dataset is bundled; no benchmark result has been claimed.
- PDF loading is optional and only enabled when its parser is installed; complex layouts and scanned documents are skipped rather than OCR'd.

