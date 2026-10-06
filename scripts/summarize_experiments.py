from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.run_retrieval_experiments import build_summary, canonical_configs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=Path, default=Path("eval/experiments/grid.json"))
    parser.add_argument("--runs", type=Path, default=Path("results/experiments/runs"))
    parser.add_argument("--output", type=Path, default=Path("results/experiments"))
    args = parser.parse_args()

    grid = json.loads(args.grid.read_text(encoding="utf-8"))
    expected = len(canonical_configs(grid))
    files = sorted(args.runs.glob("*.json"))
    summary = build_summary(files, args.output)
    summary["configs_expected_full"] = expected
    summary["complete"] = summary["configs_completed"] == expected
    (args.output / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    if not summary["complete"]:
        raise SystemExit(
            f"experiment grid incomplete: {summary['configs_completed']}/{expected} configurations"
        )
    print(f"complete experiment grid: {expected}/{expected} configurations")


if __name__ == "__main__":
    main()
