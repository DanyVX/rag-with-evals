from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from ragx.chunk.core import Chunk

BAD_MARKERS = {
    "passage",
    "question",
    "answer",
    "return only",
    "source",
}


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


def content_words(text: str) -> set[str]:
    stop = {
        "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "is", "are",
        "was", "were", "does", "do", "did", "what", "which", "how", "when", "where",
        "python", "with", "from", "by", "this", "that", "it", "be", "as", "at",
    }
    return {w for w in words(text) if len(w) > 2 and w not in stop}


def validate_item(item: dict, chunks: dict[str, Chunk]) -> list[str]:
    reasons: list[str] = []
    question = str(item.get("question") or "").strip()
    answer = str(item.get("gold_answer") or "").strip()
    qwords = words(question)
    awords = words(answer)

    if not question.endswith("?"):
        reasons.append("question_missing_question_mark")
    if not (5 <= len(qwords) <= 36):
        reasons.append("question_length")
    if len(set(qwords)) < 4:
        reasons.append("question_too_trivial")
    if any(question.lower().startswith(marker + ":") for marker in BAD_MARKERS):
        reasons.append("question_prompt_leakage")

    if item.get("answerable"):
        if not (1 <= len(awords) <= 80):
            reasons.append("answer_length")
        normalized_answer = " ".join(awords)
        if normalized_answer in BAD_MARKERS or any(
            marker + ":" in answer.lower() for marker in BAD_MARKERS
        ):
            reasons.append("answer_prompt_leakage")
        ids = item.get("gold_chunk_ids") or []
        if not ids or any(cid not in chunks for cid in ids):
            reasons.append("missing_gold_chunk")
        else:
            evidence = " ".join(chunks[cid].text for cid in ids if cid in chunks)
            evidence_words = content_words(evidence)
            answer_words = content_words(answer)
            if answer_words:
                overlap = len(answer_words & evidence_words) / len(answer_words)
                if overlap < 0.35:
                    reasons.append("answer_low_evidence_overlap")
        if item.get("question_type") == "numeric" and not re.search(r"\d", answer):
            reasons.append("numeric_without_number")
    else:
        if item.get("gold_chunk_ids"):
            reasons.append("unanswerable_has_gold_chunk")
        if "not found in the provided documents" not in answer.lower():
            reasons.append("unanswerable_bad_gold_answer")

    return sorted(set(reasons))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("chunks", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rejected", type=Path, required=True)
    parser.add_argument("--minimum", type=int, default=150)
    args = parser.parse_args()

    chunks = {
        chunk.id: chunk
        for chunk in (
            Chunk.model_validate_json(line)
            for line in args.chunks.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    items = [
        json.loads(line)
        for line in args.dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    accepted: list[dict] = []
    rejected: list[dict] = []
    seen_questions: set[str] = set()

    for item in items:
        key = " ".join(words(str(item.get("question") or "")))
        reasons = validate_item(item, chunks)
        if key in seen_questions:
            reasons.append("duplicate_question")
        if reasons:
            rejected.append({**item, "rejection_reasons": sorted(set(reasons))})
            continue
        seen_questions.add(key)
        accepted.append(item)

    if len(accepted) < args.minimum:
        counts: dict[str, int] = {}
        for item in rejected:
            for reason in item["rejection_reasons"]:
                counts[reason] = counts.get(reason, 0) + 1
        raise SystemExit(
            json.dumps(
                {
                    "error": "too_few_quality_candidates",
                    "accepted": len(accepted),
                    "minimum": args.minimum,
                    "rejected": len(rejected),
                    "rejection_counts": counts,
                },
                indent=2,
            )
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.rejected.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in accepted) + "\n",
        encoding="utf-8",
    )
    args.rejected.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in rejected) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"accepted": len(accepted), "rejected": len(rejected)}, indent=2))


if __name__ == "__main__":
    main()
