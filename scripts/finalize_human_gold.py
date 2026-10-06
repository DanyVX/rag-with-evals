from __future__ import annotations

import argparse
import csv
import json
from datetime import date
from pathlib import Path


def split_ids(value: str) -> list[str]:
    return [item for item in (part.strip() for part in value.split("|")) if item]


def parse_review_date(value: str, row_id: str) -> date:
    if not value.strip():
        raise ValueError(f"first_review_date is required for approved row: {row_id}")
    try:
        return date.fromisoformat(value.strip())
    except ValueError as exc:
        raise ValueError(
            f"first_review_date must use YYYY-MM-DD for row {row_id}"
        ) from exc


def parse_human_score(value: str, row_id: str) -> int:
    try:
        score = int(value.strip())
    except ValueError as exc:
        raise ValueError(f"human_answer_score_0_4 is required for row {row_id}") from exc
    if score not in range(5):
        raise ValueError(f"human_answer_score_0_4 must be 0..4 for row {row_id}")
    return score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("review_csv", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval/datasets/human_verified.jsonl"),
    )
    parser.add_argument(
        "--labels-output",
        type=Path,
        default=Path("eval/review/human_judge_labels.jsonl"),
    )
    args = parser.parse_args()

    items: list[dict] = []
    labels: list[dict] = []
    with args.review_csv.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            status = row["human_status"].strip().upper()
            if status not in {"APPROVED", "CORRECTED"}:
                continue
            reviewed_on = parse_review_date(row["first_review_date"], row["id"])
            human_score = parse_human_score(row["human_answer_score_0_4"], row["id"])
            question = row["human_corrected_question"].strip() or row["question"].strip()
            answer = row["human_gold_answer"].strip() or row["generated_answer"].strip()
            ids = split_ids(row["human_gold_chunk_ids"]) or split_ids(row["gold_chunk_ids"])
            sources = split_ids(row["human_gold_sources"]) or split_ids(row["gold_sources"])
            answerable = row["question_type"] != "unanswerable"
            if not question or not answer or (answerable and not (ids or sources)):
                raise ValueError(f"incomplete approved row: {row['id']}")
            items.append(
                {
                    "id": row["id"],
                    "question": question,
                    "question_type": row["question_type"],
                    "answerable": answerable,
                    "split": "human_verified",
                    "gold_chunk_ids": ids,
                    "gold_sources": sources,
                    "gold_answer": answer,
                    "ambiguous": row["ambiguous"].strip().lower() in {"1", "true", "yes"},
                    "notes": row["reviewer_notes"].strip() or None,
                }
            )
            labels.append(
                {
                    "id": row["id"],
                    "question": question,
                    "reference_answer": answer,
                    "candidate_answer": row["generated_answer"].strip(),
                    "gold_chunk_ids": ids,
                    "gold_sources": sources,
                    "human_score": human_score,
                    "first_review_date": reviewed_on.isoformat(),
                }
            )

    if len(items) < 100:
        raise ValueError(f"at least 100 approved/corrected rows required; found {len(items)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in items) + "\n",
        encoding="utf-8",
    )
    args.labels_output.parent.mkdir(parents=True, exist_ok=True)
    args.labels_output.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in labels) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(items)} human-verified items to {args.output}")
    print(f"wrote {len(labels)} human judge labels to {args.labels_output}")


if __name__ == "__main__":
    main()
