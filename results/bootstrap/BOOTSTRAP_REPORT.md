# Bootstrap evaluation report

These are observed pipeline-validation results from the pinned Python 3.11.17 documentation corpus. They are **not release headline metrics** because the candidate questions were synthetic and had not yet passed human verification.

- Python chunks: 13,791
- NIST PDF chunks: 48
- Generated evaluation items: 150
- Retrieval questions evaluated: 120
- Embedding model: BAAI/bge-small-en-v1.5

| Retriever | Recall@5 | 95% CI | MRR | 95% CI |
|---|---:|---:|---:|---:|
| BM25 | 61.25% | 53.33%-68.76% | 0.5086 | 0.4391-0.5801 |
| Dense | 43.75% | 35.42%-52.08% | 0.3802 | 0.3062-0.4538 |
| Hybrid RRF | 55.42% | 47.08%-63.33% | 0.4487 | 0.3831-0.5150 |

Prompt-injection bootstrap attack success rate: **25% (1/4)**.

The synthetic generation output from this run showed material quality defects, including mismatched generated answers and prompt-token leakage such as `PASSAGE`. Those questions are not accepted as human gold and will be regenerated behind stricter automatic quality gates.
