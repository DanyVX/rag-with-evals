# Architecture

    source files
        |
        v
    loaders -> cleaning -> dedupe -> ingest manifest
        |
        v
    chunk strategies
      fixed / recursive / sentence / structure
        |
        +----------------------+
        |                      |
        v                      v
    embeddings              BM25
        |                      |
        v                      |
    FAISS + SQLite             |
        |                      |
        +------> retrieval <---+
                   |
             RRF / MMR / floor
                   |
              cross-encoder
                   |
                   v
             prompt packing
                   |
           provider interface
          /                  \
    Anthropic        OpenAI-compatible
                   |
                   v
          citation validation
                   |
                   v
       evaluation + experiments
      retrieval / faithfulness
      abstention / cost / latency
      bootstrap CIs / permutation
                   |
                   v
        reports / heatmap / Pareto

The orchestration is intentionally explicit. Libraries provide embeddings, vector search, parsing, and HTTP primitives, but no end-to-end RAG framework controls the pipeline.
