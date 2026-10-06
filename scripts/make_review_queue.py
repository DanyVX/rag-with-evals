from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, default=Path("eval/review/human_review_queue.csv"))
    parser.add_argument("--limit", type=int, default=120)
    args = parser.parse_args()

    items = [
        json.loads(line)
        for line in args.dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "id",
        "question",
        "question_type",
        "generated_answer",
        "gold_chunk_ids",
        "gold_sources",
        "human_status",
        "human_corrected_question",
        "human_gold_answer",
        "human_gold_chunk_ids",
        "human_gold_sources",
        "human_answer_score_0_4",
        "ambiguous",
        "reviewer_notes",
        "first_review_date",
    ]
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in items:
            writer.writerow(
                {
                    "id": item["id"],
                    "question": item["question"],
                    "question_type": item["question_type"],
                    "generated_answer": item.get("gold_answer") or "",
                    "gold_chunk_ids": "|".join(item.get("gold_chunk_ids", [])),
                    "gold_sources": "|".join(item.get("gold_sources", [])),
                    "human_status": "PENDING",
                    "ambiguous": item.get("ambiguous", False),
                }
            )
    print(f"wrote {len(items)} review rows to {args.output}")


if __name__ == "__main__":
    main()
