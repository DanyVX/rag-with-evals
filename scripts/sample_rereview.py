from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

SEED = 20261006


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("review_csv", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval/review/rereview_30.csv"),
    )
    parser.add_argument("--count", type=int, default=30)
    args = parser.parse_args()

    with args.review_csv.open(encoding="utf-8", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["human_status"].strip().upper() in {"APPROVED", "CORRECTED"}
        ]
    if len(rows) < args.count:
        raise ValueError(f"need at least {args.count} completed first reviews; found {len(rows)}")

    rng = random.Random(SEED)
    sample = rng.sample(rows, args.count)
    fields = [
        "id",
        "question",
        "first_status",
        "first_answer",
        "first_gold_chunk_ids",
        "second_status",
        "second_answer",
        "second_gold_chunk_ids",
        "second_notes",
        "rereview_date",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sample:
            writer.writerow(
                {
                    "id": row["id"],
                    "question": row["human_corrected_question"].strip() or row["question"],
                    "first_status": row["human_status"],
                    "first_answer": row["human_gold_answer"].strip() or row["generated_answer"],
                    "first_gold_chunk_ids": row["human_gold_chunk_ids"].strip()
                    or row["gold_chunk_ids"],
                }
            )
    print(f"wrote deterministic {args.count}-item re-review sample to {args.output}")


if __name__ == "__main__":
    main()
