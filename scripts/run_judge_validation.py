from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from ragx.chunk.core import Chunk
from ragx.eval.judge import cohens_kappa

MODEL_ID = "google/flan-t5-base"
RUBRIC = """Score the candidate answer from 0 to 4.
4 = fully correct and supported by the reference/evidence.
3 = mostly correct, minor omission.
2 = partially correct, meaningful error or omission.
1 = mostly incorrect but contains a small correct element.
0 = incorrect, unsupported, or should have abstained.
Do not reward verbosity or writing style. Return only one digit: 0, 1, 2, 3, or 4.
"""


def load_chunks(path: Path) -> dict[str, Chunk]:
    return {
        chunk.id: chunk
        for chunk in (
            Chunk.model_validate_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def cache_key(item: dict) -> str:
    payload = json.dumps(item, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def parse_score(text: str) -> int | None:
    match = re.search(r"\b([0-4])\b", text)
    return int(match.group(1)) if match else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("labels", type=Path)
    parser.add_argument("chunks", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/final/judge_validation.json"),
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("cache/judge_validation.json"),
    )
    parser.add_argument("--minimum", type=int, default=30)
    args = parser.parse_args()

    labels = [
        json.loads(line)
        for line in args.labels.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(labels) < args.minimum:
        raise ValueError(f"need at least {args.minimum} human labels; found {len(labels)}")

    chunks = load_chunks(args.chunks)
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    cache = (
        json.loads(args.cache.read_text(encoding="utf-8"))
        if args.cache.exists()
        else {}
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)

    human: list[int] = []
    judge: list[int] = []
    details: list[dict] = []
    judge_errors = 0

    for item in labels:
        evidence = "\n\n".join(
            chunks[cid].text for cid in item["gold_chunk_ids"] if cid in chunks
        )
        prompt = (
            RUBRIC
            + f"\nQUESTION:\n{item['question']}\n"
            + f"REFERENCE ANSWER:\n{item['reference_answer']}\n"
            + f"EVIDENCE:\n{evidence[:5000]}\n"
            + f"CANDIDATE ANSWER:\n{item['candidate_answer']}\n"
        )
        key = cache_key(
            {
                "model": MODEL_ID,
                "rubric": RUBRIC,
                "question": item["question"],
                "reference": item["reference_answer"],
                "evidence": evidence[:5000],
                "candidate": item["candidate_answer"],
            }
        )
        if key in cache:
            raw = cache[key]
        else:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            output = model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
                num_beams=1,
            )
            raw = tokenizer.decode(output[0], skip_special_tokens=True).strip()
            cache[key] = raw

        score = parse_score(raw)
        if score is None:
            judge_errors += 1
            details.append({"id": item["id"], "human_score": item["human_score"], "judge_error": raw})
            continue
        human.append(int(item["human_score"]))
        judge.append(score)
        details.append(
            {
                "id": item["id"],
                "human_score": int(item["human_score"]),
                "judge_score": score,
                "raw": raw,
            }
        )

    args.cache.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")
    if len(human) < args.minimum:
        raise ValueError(
            f"fewer than {args.minimum} parseable judge results after errors: {len(human)}"
        )

    exact = sum(a == b for a, b in zip(human, judge)) / len(human)
    within_one = sum(abs(a - b) <= 1 for a, b in zip(human, judge)) / len(human)
    result = {
        "judge_model": MODEL_ID,
        "n": len(human),
        "judge_errors": judge_errors,
        "exact_agreement": exact,
        "within_one_agreement": within_one,
        "cohens_kappa": cohens_kappa(human, judge),
        "known_biases": [
            "verbosity preference",
            "position effects",
            "reference-answer anchoring",
            "small-model rubric parsing noise",
        ],
        "details": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "details"}, indent=2))


if __name__ == "__main__":
    main()
