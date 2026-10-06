from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--failures",
        type=Path,
        default=Path("results/final/failure_cases.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/final/FAILURE_ANALYSIS.md"),
    )
    args = parser.parse_args()

    failures = json.loads(args.failures.read_text(encoding="utf-8"))
    counts = Counter(
        category
        for item in failures
        for category in item.get("categories", [])
    )
    categories = [
        "retrieval_miss",
        "chunk_split",
        "wrong_reranking",
        "unsupported_claim",
        "citation_failure",
        "wrong_abstention",
        "extraction_failure",
        "prompt_injection",
        "judge_error",
    ]

    lines = [
        "# Final failure analysis",
        "",
        f"Total failed/question-level flagged cases: {len(failures)}",
        "",
        "## Counts",
        "",
        "| Category | Count |",
        "|---|---:|",
    ]
    for category in categories:
        lines.append(f"| {category} | {counts.get(category, 0)} |")

    lines.extend(["", "## Representative cases", ""])
    for category in categories:
        examples = [
            item for item in failures if category in item.get("categories", [])
        ][:3]
        lines.append(f"### {category}")
        if not examples:
            lines.append("")
            lines.append("No observed cases in this run.")
            lines.append("")
            continue
        for item in examples:
            lines.extend(
                [
                    "",
                    f"- ID: {item['id']}",
                    f"  - Answer: {str(item.get('answer', ''))[:500]}",
                    f"  - Gold: {str(item.get('gold_answer', ''))[:500]}",
                    "  - Retrieved sources: "
                    + ", ".join(item.get("retrieved_sources", [])[:5]),
                ]
            )
        lines.append("")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
