from __future__ import annotations

import json
from pathlib import Path

from ragx.eval.dataset import load_jsonl, validate_human_subset

REQUIRED = [
    Path("corpus/checksums.json"),
    Path("eval/datasets/human_verified.jsonl"),
    Path("eval/review/human_judge_labels.jsonl"),
    Path("results/final/metrics.json"),
    Path("results/final/injection_eval.json"),
    Path("results/final/judge_validation.json"),
    Path("results/final/FAILURE_ANALYSIS.md"),
    Path("results/experiments/experiment_summary.json"),
    Path("results/experiments/compact_results.jsonl"),
    Path("results/human/rereview_consistency.json"),
]


def require_ci(metric: dict, name: str) -> None:
    if int(metric.get("n", 0)) <= 0:
        raise SystemExit(f"release gate failed; {name} has no observations")
    ci = metric.get("ci95")
    if not isinstance(ci, list) or len(ci) != 2:
        raise SystemExit(f"release gate failed; {name} is missing a 95% CI")


def main() -> None:
    missing = [str(path) for path in REQUIRED if not path.exists()]
    if missing:
        raise SystemExit("release gate failed; missing: " + ", ".join(missing))

    items = load_jsonl("eval/datasets/human_verified.jsonl")
    validate_human_subset(items)

    metrics = json.loads(Path("results/final/metrics.json").read_text(encoding="utf-8"))
    human = metrics.get("human_verified") or {}
    if int(human.get("question_count", 0)) < 100:
        raise SystemExit("release gate failed; final metrics contain fewer than 100 human questions")
    if not metrics.get("synthetic_retrieval"):
        raise SystemExit("release gate failed; synthetic and human results are not reported separately")
    require_ci(human.get("retrieval", {}).get("mrr", {}), "human retrieval MRR")
    require_ci(human.get("retrieval", {}).get("recall@5", {}), "human retrieval Recall@5")
    require_ci(human.get("generation", {}).get("correctness", {}), "generation correctness")
    require_ci(human.get("generation", {}).get("faithfulness", {}), "generation faithfulness")
    require_ci(
        human.get("generation", {}).get("citation_precision", {}),
        "citation precision",
    )
    require_ci(
        human.get("generation", {}).get("citation_recall", {}),
        "citation recall",
    )

    experiments = json.loads(
        Path("results/experiments/experiment_summary.json").read_text(encoding="utf-8")
    )
    if not experiments.get("complete"):
        raise SystemExit("release gate failed; controlled experiment grid is incomplete")
    if int(experiments.get("configs_completed", 0)) != int(
        experiments.get("configs_expected_full", -1)
    ):
        raise SystemExit("release gate failed; experiment result count does not match full grid")
    compact_lines = [
        line
        for line in Path("results/experiments/compact_results.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    if len(compact_lines) != int(experiments["configs_expected_full"]):
        raise SystemExit("release gate failed; compact experiment evidence is incomplete")

    judge = json.loads(
        Path("results/final/judge_validation.json").read_text(encoding="utf-8")
    )
    if int(judge.get("n", 0)) < 30 or "cohens_kappa" not in judge:
        raise SystemExit("release gate failed; judge is not validated against >=30 human labels")

    injection = json.loads(
        Path("results/final/injection_eval.json").read_text(encoding="utf-8")
    )
    if int(injection.get("total", 0)) <= 0 or "attack_success_rate" not in injection:
        raise SystemExit("release gate failed; prompt-injection results are incomplete")

    rereview = json.loads(
        Path("results/human/rereview_consistency.json").read_text(encoding="utf-8")
    )
    if int(rereview.get("n", 0)) < 30:
        raise SystemExit("release gate failed; fewer than 30 completed re-reviews")
    if int(rereview.get("minimum_gap_days", 0)) < 7:
        raise SystemExit("release gate failed; re-review interval is shorter than seven days")

    failure_text = Path("results/final/FAILURE_ANALYSIS.md").read_text(encoding="utf-8")
    if "Representative cases" not in failure_text:
        raise SystemExit("release gate failed; final failure analysis lacks observed examples")

    print("release gate passed for v0.1.0")


if __name__ == "__main__":
    main()
