# Failure analysis

This file records failure categories, not invented examples or measurements.

| Category | Detection | Current mitigation | Remaining risk |
|---|---|---|---|
| Retrieval miss | Gold chunk absent from top-k | BM25 baseline and evaluation runner | Dense/hybrid retrieval is not available in this CPU-only baseline. |
| Chunk boundary | Answer text split across chunks | Configurable overlap and stable chunk IDs | Only fixed-token chunking is currently implemented. |
| Wrong abstention | No evidence but non-abstaining answer | Offline path is evidence-only and abstains on zero hits | Remote providers require separate adapter-level evaluation. |
| Prompt injection | Instructions found in retrieved content | Sources are delimited as untrusted data | Delimiters reduce but do not eliminate LLM instruction-following risk. |
| Extraction failure | Empty/non-substantive input or unsupported parser | Skip with a visible warning | PDF parser is an optional dependency and layout quality needs corpus-specific tests. |
