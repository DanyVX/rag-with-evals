# Failure analysis

Every evaluation run should categorize failed questions into one primary failure class and optional secondary classes.

## Retrieval miss
The gold evidence was not present in the final retrieved set. Record whether dense, BM25, hybrid, MMR, or reranking caused the miss.

## Chunk split
The necessary answer span crossed a chunk boundary or a table/code block was damaged by chunking. Compare overlap and structure-aware configurations.

## Wrong reranking
A relevant candidate was retrieved in top-N but pushed below final top-k by the cross-encoder.

## Hallucination / unsupported claim
Generation produced a factual claim unsupported by cited context. Record claim-level groundedness and the cited chunk.

## Citation failure
The answer used a malformed, nonexistent, or irrelevant citation.

## Wrong abstention
False refusal: evidence existed but the model abstained. False answer: no evidence existed but the model answered anyway.

## Extraction failure
Important source content was lost or garbled during HTML/PDF extraction. Flag scans, low-alphabetic table output, repeated headers/footers, encoding problems, and layout errors.

## Prompt injection
Retrieved content changed model behavior or caused instruction following from within the source. Record attack type and whether output validation caught it.

## Judge error
The evaluation judge returned malformed output after retry or disagreed materially with human labels. Judge errors are reported separately and never silently dropped.
