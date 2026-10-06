from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def split_ids(value: str) -> list[str]:
    return [item for item in (part.strip() for part in value.split("|")) if item]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("review_csv", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval/datasets/human_verified.jsonl"),
    )
    args = parser.parse_args()

    items: list[dict] = []
    with args.review_csv.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            status = row["human_status"].strip().upper()
            if status not in {"APPROVED", "CORRECTED"}:
                continue
            question = row["human_corrected_question"].strip() or row["question"].strip()
            answer = row["human_gold_answer"].strip() or row["generated_answer"].strip()
            ids = split_ids(row["human_gold_chunk_ids"]) or split_ids(row["gold_chunk_ids"])
            answerable = row["question_type"] != "unanswerable"
            if not question or not answer or (answerable and not ids):
                raise ValueError(f"incomplete approved row: {row['id']}")
            items.append(
                {
                    "id": row["id"],
                    "question": question,
                    "question_type": row["question_type"],
                    "answerable": answerable,
                    "split": "human_verified",
                    "gold_chunk_ids": ids,
                    "gold_answer": answer,
                    "ambiguous": row["ambiguous"].strip().lower() in {"1", "true", "yes"},
                    "notes": row["reviewer_notes"].strip() or None,
                }
            )

    if len(items) < 100:
        raise ValueError(f"at least 100 approved/corrected rows required; found {len(items)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in items) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(items)} human-verified items to {args.output}")


if __name__ == "__main__":
    main()
