from __future__ import annotations

import argparse
import json
from pathlib import Path

START = "<!-- RAGX_RESULTS_START -->"
END = "<!-- RAGX_RESULTS_END -->"


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.2f}%"


def metric_line(label: str, metric: dict) -> str:
    mean = metric.get("mean")
    ci = metric.get("ci95")
    if mean is None or not ci:
        return f"| {label} | n/a | n/a | {metric.get('n', 0)} |"
    return (
        f"| {label} | {pct(float(mean))} | "
        f"{pct(float(ci[0]))}–{pct(float(ci[1]))} | {metric.get('n', 0)} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    parser.add_argument("--metrics", type=Path, default=Path("results/final/metrics.json"))
    parser.add_argument(
        "--injection",
        type=Path,
        default=Path("results/final/injection_eval.json"),
    )
    parser.add_argument(
        "--judge",
        type=Path,
        default=Path("results/final/judge_validation.json"),
    )
    parser.add_argument(
        "--rereview",
        type=Path,
        default=Path("results/human/rereview_consistency.json"),
    )
    args = parser.parse_args()

    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    injection = json.loads(args.injection.read_text(encoding="utf-8"))
    judge = json.loads(args.judge.read_text(encoding="utf-8"))
    human = metrics["human_verified"]

    lines = [
        f"Human-verified questions: **{human['question_count']}**. "
        "Synthetic and human results are reported separately.",
        "",
        "| Metric | Mean | 95% bootstrap CI | n |",
        "|---|---:|---:|---:|",
        metric_line("Retrieval MRR", human["retrieval"]["mrr"]),
        metric_line("Retrieval Recall@5", human["retrieval"]["recall@5"]),
        metric_line("Answer correctness", human["generation"]["correctness"]),
        metric_line("Faithfulness", human["generation"]["faithfulness"]),
        metric_line("Citation precision", human["generation"]["citation_precision"]),
        metric_line("Citation recall", human["generation"]["citation_recall"]),
        "",
        f"Prompt-injection attack success rate: **{pct(injection['attack_success_rate'])}** "
        f"({injection['successful_attacks']}/{injection['total']}).",
        "",
        f"Judge validation: **n={judge['n']}**, exact agreement "
        f"**{pct(judge['exact_agreement'])}**, Cohen's kappa "
        f"**{judge['cohens_kappa']:.3f}**.",
    ]

    if args.rereview.exists():
        rereview = json.loads(args.rereview.read_text(encoding="utf-8"))
        lines.extend(
            [
                "",
                f"One-week re-review consistency: **n={rereview['n']}**, "
                f"exact answer agreement **{pct(rereview['exact_answer_agreement'])}**, "
                f"gold-chunk-set agreement **{pct(rereview['gold_chunk_set_agreement'])}**.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "The required 30-item one-week re-review is still pending; release remains blocked.",
            ]
        )

    replacement = START + "\n" + "\n".join(lines) + "\n" + END
    text = args.readme.read_text(encoding="utf-8")
    if START not in text or END not in text:
        raise ValueError("README result markers are missing")
    before, rest = text.split(START, 1)
    _, after = rest.split(END, 1)
    args.readme.write_text(before + replacement + after, encoding="utf-8")


if __name__ == "__main__":
    main()
