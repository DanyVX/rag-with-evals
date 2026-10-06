# Evaluation dataset datasheet

## Purpose
Evaluate retrieval quality, generation correctness, groundedness, citations, abstention, and system cost/latency.

## Required layers
1. Synthetic questions generated from sampled evidence chunks.
2. Plausible but corpus-absent unanswerable questions.
3. A human-verified gold subset of at least 100 questions.

## Human verification
For each human-verified item, confirm the question, gold chunk(s), and gold answer manually. Re-review a sample of 30 items after at least one week and record consistency.

## Bias controls
Synthetic and human-verified results must be reported separately. Synthetic wording must be paraphrased away from source wording to reduce lexical leakage toward BM25.

## Versioning
Each published dataset snapshot must include a checksum manifest and immutable version identifier.

## Ambiguity
Questions with multiple valid answers, negation, time sensitivity, or ambiguous wording must be labeled rather than silently forced into a single gold.
