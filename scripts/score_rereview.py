from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ragx.eval.judge import cohens_kappa


def normalize(value: str) -> str:
    return " ".join(value.lower().split())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("rereview_csv", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/human/rereview_consistency.json"),
    )
    args = parser.parse_args()

    labels_a: list[int] = []
    labels_b: list[int] = []
    answer_agreement: list[bool] = []
    chunk_agreement: list[bool] = []
    rows = []
    with args.rereview_csv.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if not row["second_status"].strip():
                continue
            first_ok = row["first_status"].strip().upper() in {"APPROVED", "CORRECTED"}
            second_ok = row["second_status"].strip().upper() in {"APPROVED", "CORRECTED"}
            labels_a.append(int(first_ok))
            labels_b.append(int(second_ok))
            answer_agreement.append(
                normalize(row["first_answer"]) == normalize(row["second_answer"])
            )
            chunk_agreement.append(
                set(filter(None, row["first_gold_chunk_ids"].split("|")))
                == set(filter(None, row["second_gold_chunk_ids"].split("|")))
            )
            rows.append(row["id"])

    if len(rows) < 30:
        raise ValueError(f"30 completed re-reviews required; found {len(rows)}")
    result = {
        "n": len(rows),
        "status_kappa": cohens_kappa(labels_a, labels_b),
        "exact_answer_agreement": sum(answer_agreement) / len(answer_agreement),
        "gold_chunk_set_agreement": sum(chunk_agreement) / len(chunk_agreement),
        "ids": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
