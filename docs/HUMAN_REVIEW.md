# Human verification procedure

The bootstrap workflow produces eval/review/human_review_queue.csv with 120 candidate questions. At least 100 must be genuinely reviewed by a person before human_verified.jsonl can be created.

## Review each row

Use the source chunk IDs and generated material to check the question against the corpus.

Fill these columns:

- human_status
  - APPROVED: question, evidence, and gold answer are correct as written.
  - CORRECTED: the item is usable after editing the question, answer, and/or gold chunk IDs.
  - REJECTED: ambiguous, unsupported, broken, duplicated, or otherwise unsuitable.
- human_corrected_question: required only when the question needs correction.
- human_gold_answer: enter the verified answer. It may match the generated answer.
- human_gold_chunk_ids: enter verified chunk IDs separated with |. Unanswerable items may leave this empty.
- human_answer_score_0_4: score the original generated answer against the verified gold:
  - 4 fully correct
  - 3 mostly correct, minor omission
  - 2 partially correct
  - 1 mostly incorrect with a small correct element
  - 0 incorrect/unsupported or should have abstained
- ambiguous: true if the wording admits multiple reasonable interpretations.
- reviewer_notes: record meaningful corrections or rejection reasons.
- first_review_date: ISO date YYYY-MM-DD.

Do not mark an item approved merely because it sounds plausible. Verify it against the cited chunk(s).

## Finalize the gold set

After at least 100 rows are APPROVED or CORRECTED:

    python scripts/finalize_human_gold.py eval/review/human_review_queue.csv

This creates:

- eval/datasets/human_verified.jsonl
- eval/review/human_judge_labels.jsonl

The command refuses to finalize fewer than 100 items or approved rows missing dates, scores, answers, or evidence.

## Judge validation

The same human 0–4 scores are used to validate the independent judge:

    python scripts/run_judge_validation.py eval/review/human_judge_labels.jsonl data/processed/python_chunks.jsonl

The judge is google/flan-t5-base, different from the bootstrap generator google/flan-t5-small. Results include exact agreement, within-one agreement, judge errors, and Cohen's kappa.

## One-week consistency check

Create the deterministic 30-item sample after the first review:

    python scripts/sample_rereview.py eval/review/human_review_queue.csv

Do not fill the second review immediately. At least seven calendar days after each item's first_review_date, independently fill:

- second_status
- second_answer
- second_gold_chunk_ids
- second_notes
- rereview_date

Then run:

    python scripts/score_rereview.py eval/review/rereview_30.csv

The scorer rejects re-reviews performed fewer than seven days after the original review.
