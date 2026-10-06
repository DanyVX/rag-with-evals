# rag-with-evals

A modular Retrieval-Augmented Generation system built as a **measurement instrument**, not a chat demo.

## Goals
- Measure retrieval quality, generation faithfulness, citation quality, abstention, latency, and cost.
- Run controlled experiments over chunking, retrievers, rerankers, and embedding models.
- Report confidence intervals and distinguish synthetic from human-verified evaluation results.
- Keep the RAG pipeline explicit and understandable rather than hiding it behind an end-to-end framework.

## Status
Work in progress. M0 setup is being implemented first.

## Planned CLI
```bash
ragx ingest
ragx index
ragx ask
ragx eval
ragx report
```

## Safety / cost
Real provider calls are disabled until `RAGX_MAX_SPEND_USD` is explicitly set to a positive value.
