from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_results(results_dir: str | Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(Path(results_dir).glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") == "ok":
            rows.append({"config": payload["config"], **payload["result"]})
    return rows


def save_heatmap(
    matrix: np.ndarray,
    x_labels: list[str],
    y_labels: list[str],
    *,
    title: str,
    output: str | Path,
) -> None:
    fig, ax = plt.subplots()
    image = ax.imshow(matrix, aspect="auto")
    ax.set_xticks(range(len(x_labels)), labels=x_labels)
    ax.set_yticks(range(len(y_labels)), labels=y_labels)
    ax.set_title(title)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def save_pareto(
    rows: list[dict],
    *,
    quality_key: str,
    cost_key: str,
    output: str | Path,
) -> None:
    xs = [float(r[cost_key]) for r in rows]
    ys = [float(r[quality_key]) for r in rows]
    fig, ax = plt.subplots()
    ax.scatter(xs, ys)
    ax.set_xlabel(cost_key)
    ax.set_ylabel(quality_key)
    ax.set_title("Quality vs cost/latency")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def effect_summary(rows: list[dict], metric: str, factor: str) -> list[tuple[str, float]]:
    groups: dict[str, list[float]] = {}
    for row in rows:
        value = str(row["config"].get(factor))
        groups.setdefault(value, []).append(float(row[metric]))
    means = [(value, sum(vals) / len(vals)) for value, vals in groups.items()]
    return sorted(means, key=lambda item: item[1], reverse=True)
