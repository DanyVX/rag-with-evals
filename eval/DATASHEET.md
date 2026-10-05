# Evaluation datasheet

Evaluation records are JSON objects with `id`, `question`, `gold_chunk_ids`, `answerable`, `question_type`, and optional `gold_answer`. Synthetic and human-verified records must be reported separately. The committed test fixture is not a benchmark and must never be reported as one.

## Governance

Use public, license-recorded sources only. Do not include personal data. Mark ambiguous, time-sensitive, multi-hop, and unanswerable questions explicitly. Results must include sample counts and bootstrap confidence intervals.
