# Project decisions

## Architecture
The RAG pipeline is implemented explicitly. End-to-end wrappers such as LangChain and LlamaIndex are intentionally not used for orchestration.

## Vector store
Initial implementation target: FAISS + SQLite behind a VectorStore interface. A second implementation can be added later without changing retrieval code.

## Provider access
Provider integrations will sit behind a common interface. Real provider calls are disabled by default until RAGX_MAX_SPEND_USD is set to a positive value.

## Corpus
Pending owner decision. The selected primary corpus must be license-clear and large enough to create thousands of chunks. A second small PDF/table-heavy corpus will exercise extraction failure cases.

## Evaluation principle
Synthetic and human-verified results are reported separately. Headline comparisons require confidence intervals and paired statistical testing.
