from __future__ import annotations

import json
from pathlib import Path

from ragx.eval.dataset import load_jsonl, validate_human_subset

REQUIRED = [
    Path("corpus/checksums.json"),
    Path("eval/datasets/human_verified.jsonl"),
    Path("results/final/metrics.json"),
    Path("results/final/injection_eval.json"),
    Path("results/final/judge_validation.json"),
    Path("results/human/rereview_consistency.json"),
    Path("docs/FAILURE_ANALYSIS.md"),
]


def main() -> None:
    missing = [str(path) for path in REQUIRED if not path.exists()]
    if missing:
        raise SystemExit("release gate failed; missing: " + ", ".join(missing))

    items = load_jsonl("eval/datasets/human_verified.jsonl")
    validate_human_subset(items)

    rereview = json.loads(
        Path("results/human/rereview_consistency.json").read_text(encoding="utf-8")
    )
    if int(rereview.get("n", 0)) < 30:
        raise SystemExit("release gate failed; fewer than 30 completed re-reviews")

    print("release gate passed for v0.1.0")


if __name__ == "__main__":
    main()
