from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from ragx.chunk.core import Chunk

SEED = 17
MODEL_ID = "google/flan-t5-small"


def load_chunks(path: Path) -> list[Chunk]:
    return [
        Chunk.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def has_four_word_overlap(question: str, passage: str) -> bool:
    def grams(text: str) -> set[tuple[str, ...]]:
        words = re.findall(r"[a-z0-9_]+", text.lower())
        return {tuple(words[i : i + 4]) for i in range(max(0, len(words) - 3))}
    return bool(grams(question) & grams(passage))


class LocalGenerator:
    def __init__(self, model_id: str = MODEL_ID) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    def generate(self, prompt: str, max_new_tokens: int = 72) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=2,
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True).strip()


def make_single(generator: LocalGenerator, chunk: Chunk, qtype: str, item_id: str) -> dict:
    passage = chunk.text[:2200]
    instruction = {
        "factoid": "Write one concise factual question",
        "numeric": "Write one question whose answer contains a number, quantity, or version",
        "list": "Write one question whose answer is a short list",
        "single_hop": "Write one conceptual question",
    }[qtype]
    question = generator.generate(
        f"{instruction} that can be answered from the passage. "
        "Paraphrase aggressively and do not copy a phrase of four or more words. "
        f"Return only the question.\n\nPASSAGE:\n{passage}"
    )
    if not question.endswith("?"):
        question = question.rstrip(".") + "?"
    if has_four_word_overlap(question, passage):
        question = generator.generate(
            "Paraphrase this question so it keeps the same meaning but avoids copying "
            "four-word phrases from the passage. Return only the rewritten question.\n\n"
            f"QUESTION:\n{question}\n\nPASSAGE:\n{passage}"
        )
        if not question.endswith("?"):
            question = question.rstrip(".") + "?"
    answer = generator.generate(
        "Answer the question using only the passage. Be concise and factual. "
        "Return only the answer.\n\n"
        f"QUESTION:\n{question}\n\nPASSAGE:\n{passage}",
        max_new_tokens=96,
    )
    return {
        "id": item_id,
        "question": question,
        "question_type": qtype,
        "answerable": True,
        "split": "synthetic",
        "gold_chunk_ids": [chunk.id],
        "gold_sources": [chunk.source],
        "gold_answer": answer,
        "ambiguous": False,
        "notes": f"Generated locally with {MODEL_ID}; requires human verification before gold use.",
    }


def make_multihop(
    generator: LocalGenerator,
    a: Chunk,
    b: Chunk,
    item_id: str,
) -> dict:
    pa, pb = a.text[:1200], b.text[:1200]
    question = generator.generate(
        "Create one question that requires combining one fact from Passage A and one fact "
        "from Passage B. Paraphrase rather than copying source wording. Return only the question.\n\n"
        f"PASSAGE A:\n{pa}\n\nPASSAGE B:\n{pb}"
    )
    if not question.endswith("?"):
        question = question.rstrip(".") + "?"
    answer = generator.generate(
        "Answer using both passages. Return only the answer.\n\n"
        f"QUESTION:\n{question}\n\nPASSAGE A:\n{pa}\n\nPASSAGE B:\n{pb}",
        max_new_tokens=96,
    )
    return {
        "id": item_id,
        "question": question,
        "question_type": "multi_hop",
        "answerable": True,
        "split": "synthetic",
        "gold_chunk_ids": [a.id, b.id],
        "gold_sources": sorted({a.source, b.source}),
        "gold_answer": answer,
        "ambiguous": False,
        "notes": f"Generated locally with {MODEL_ID}; requires human verification before gold use.",
    }


def make_unanswerable(generator: LocalGenerator, token: str, item_id: str) -> dict:
    question = generator.generate(
        "Write one plausible Python documentation question about the invented API identifier "
        f"'{token}'. The identifier does not exist. Do not mention that it is invented. "
        "Return only the question."
    )
    if token.lower() not in question.lower():
        question = f"What does '{token}' do in Python 3.11?"
    if not question.endswith("?"):
        question = question.rstrip(".") + "?"
    return {
        "id": item_id,
        "question": question,
        "question_type": "unanswerable",
        "answerable": False,
        "split": "synthetic",
        "gold_chunk_ids": [],
        "gold_sources": [],
        "gold_answer": "not found in the provided documents",
        "ambiguous": False,
        "notes": f"Invented API token {token}; requires corpus absence check and human verification.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("chunks", type=Path)
    parser.add_argument("--output", type=Path, default=Path("eval/datasets/synthetic.generated.jsonl"))
    parser.add_argument("--answerable", type=int, default=100)
    parser.add_argument("--multihop", type=int, default=20)
    parser.add_argument("--unanswerable", type=int, default=30)
    args = parser.parse_args()

    rng = random.Random(SEED)
    chunks = [
        c for c in load_chunks(args.chunks)
        if 80 <= len(c.text.split()) <= 500 and len(c.text) <= 3500
    ]
    if len(chunks) < args.answerable + args.multihop * 2:
        raise RuntimeError(f"not enough eligible chunks: {len(chunks)}")

    generator = LocalGenerator()
    rng.shuffle(chunks)
    singles = chunks[: args.answerable]
    pair_pool = chunks[args.answerable : args.answerable + args.multihop * 2]

    numeric = [c for c in singles if re.search(r"\b\d+(?:\.\d+)?\b", c.text)]
    listish = [c for c in singles if "\n-" in c.text or c.text.count(",") >= 3]
    numeric_ids = {c.id for c in numeric[: max(1, args.answerable // 5)]}
    list_ids = {c.id for c in listish[: max(1, args.answerable // 5)]}

    items: list[dict] = []
    for i, chunk in enumerate(singles, 1):
        if chunk.id in numeric_ids:
            qtype = "numeric"
        elif chunk.id in list_ids:
            qtype = "list"
        elif i % 2:
            qtype = "factoid"
        else:
            qtype = "single_hop"
        items.append(make_single(generator, chunk, qtype, f"syn-{i:04d}"))

    for i in range(args.multihop):
        a = pair_pool[i * 2]
        b = pair_pool[i * 2 + 1]
        items.append(make_multihop(generator, a, b, f"multi-{i + 1:04d}"))

    all_text = "\n".join(c.text.lower() for c in chunks)
    invented = []
    for i in range(args.unanswerable):
        token = f"sys.runtime_snapshot_cache_{i:02d}"
        if token.lower() in all_text:
            raise RuntimeError(f"invented token unexpectedly occurs in corpus: {token}")
        invented.append(token)
    for i, token in enumerate(invented, 1):
        items.append(make_unanswerable(generator, token, f"unans-{i:04d}"))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in items) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "model": MODEL_ID,
        "seed": SEED,
        "eligible_chunks": len(chunks),
        "items": len(items),
        "answerable": sum(x["answerable"] for x in items),
        "unanswerable": sum(not x["answerable"] for x in items),
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
