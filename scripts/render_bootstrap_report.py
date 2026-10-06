from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retrieval",
        type=Path,
        default=Path("results/bootstrap/retrieval_baseline.json"),
    )
    parser.add_argument(
        "--injection",
        type=Path,
        default=Path("results/bootstrap/injection_eval.json"),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("results/bootstrap/BOOTSTRAP_REPORT.md"),
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=Path("results/bootstrap/retrieval_baseline.png"),
    )
    args = parser.parse_args()

    retrieval = json.loads(args.retrieval.read_text(encoding="utf-8"))
    injection = json.loads(args.injection.read_text(encoding="utf-8"))

    names = list(retrieval["retrievers"])
    recall5 = [retrieval["retrievers"][name]["recall@5"]["mean"] for name in names]
    mrr = [retrieval["retrievers"][name]["mrr"]["mean"] for name in names]

    fig, ax = plt.subplots()
    x = list(range(len(names)))
    width = 0.35
    ax.bar([i - width / 2 for i in x], recall5, width=width, label="Recall@5")
    ax.bar([i + width / 2 for i in x], mrr, width=width, label="MRR")
    ax.set_xticks(x, labels=names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Bootstrap retrieval baseline")
    ax.legend()
    fig.tight_layout()
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot, dpi=160)
    plt.close(fig)

    lines = [
        "# Bootstrap evaluation report",
        "",
        "> These results use the generated synthetic candidate set, not the required human-verified gold subset.",
        "",
        f"Questions evaluated: {retrieval['question_count']}",
        f"Embedding model: {retrieval['embedding_model']}",
        "",
        "| Retriever | Recall@5 | 95% CI | MRR | 95% CI |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in names:
        data = retrieval["retrievers"][name]
        r = data["recall@5"]
        m = data["mrr"]
        lines.append(
            f"| {name} | {pct(r['mean'])} | {pct(r['ci95'][0])}-{pct(r['ci95'][1])} | "
            f"{m['mean']:.4f} | {m['ci95'][0]:.4f}-{m['ci95'][1]:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Prompt injection",
            "",
            f"Attack success rate: {pct(injection['attack_success_rate'])} "
            f"({injection['successful_attacks']}/{injection['total']}).",
            "",
            "## Interpretation guardrail",
            "",
            "Do not treat these as release headline numbers until the human-verified subset is complete. "
            "This bootstrap run exists to validate the pipeline and expose obvious retrieval/injection failures.",
        ]
    )
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.markdown.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
