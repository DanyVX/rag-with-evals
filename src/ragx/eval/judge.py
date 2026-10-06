from __future__ import annotations

import json
from dataclasses import dataclass

from ragx.generate.providers import LLMProvider


@dataclass(slots=True)
class JudgeResult:
    score: int | None
    rationale: str
    error: str | None = None


RUBRIC = """You are an evaluator, not the answer generator.
Score correctness from 0 to 4 using only the reference answer and supplied evidence.
Do not reward verbosity, style, or model identity.
Return strict JSON with keys score and rationale.
"""


def judge_answer(
    *,
    provider: LLMProvider,
    question: str,
    reference_answer: str,
    evidence: str,
    candidate_answer: str,
) -> JudgeResult:
    prompt = (
        RUBRIC
        + f"\nQUESTION:\n{question}\nREFERENCE:\n{reference_answer}\n"
        + f"EVIDENCE:\n{evidence}\nCANDIDATE:\n{candidate_answer}"
    )
    for _ in range(2):
        raw = provider.generate(prompt, temperature=0.0, seed=0)
        try:
            data = json.loads(raw)
            score = int(data["score"])
            if score not in range(5):
                raise ValueError("score outside 0..4")
            return JudgeResult(score=score, rationale=str(data.get("rationale", "")))
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            continue
    return JudgeResult(score=None, rationale="", error="judge_error")


def cohens_kappa(a: list[int], b: list[int]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("labels must be non-empty and equal length")
    labels = sorted(set(a) | set(b))
    observed = sum(x == y for x, y in zip(a, b)) / len(a)
    pa = {label: a.count(label) / len(a) for label in labels}
    pb = {label: b.count(label) / len(b) for label in labels}
    expected = sum(pa[label] * pb[label] for label in labels)
    return 1.0 if expected == 1.0 else (observed - expected) / (1.0 - expected)
