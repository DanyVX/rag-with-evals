from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.special import softmax
from sentence_transformers import CrossEncoder
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from ragx.chunk.core import Chunk
from ragx.embed.sentence_transformer import SentenceTransformerEmbedder
from ragx.eval.metrics import abstention_metrics, bootstrap_ci, retrieval_metrics
from ragx.generate.citations import parse_citations, validate_citations
from ragx.generate.prompt import ABSTENTION, build_prompt
from ragx.index.bm25 import BM25Index
from ragx.index.memory import InMemoryVectorStore
from ragx.retrieve.core import dense_search, hybrid_rrf, sparse_search

GENERATOR_ID = "google/flan-t5-small"
JUDGE_ID = "google/flan-t5-base"
NLI_ID = "cross-encoder/nli-deberta-v3-small"
KS = (1, 3, 5, 10, 20)


def load_chunks(path: Path) -> list[Chunk]:
    return [
        Chunk.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_items(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9.]+", text.lower()))


def numeric_values(text: str) -> set[str]:
    return set(re.findall(r"[-+]?\d+(?:\.\d+)?", text.replace(",", "")))


def split_claims(answer: str) -> list[str]:
    cleaned = re.sub(r"\[\d+\]", "", answer)
    return [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+|\n+", cleaned)
        if part.strip() and ABSTENTION not in part.lower()
    ]


class LocalSeq2Seq:
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    def generate(self, prompt: str, max_new_tokens: int) -> tuple[str, int, int]:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
        )
        text = self.tokenizer.decode(output[0], skip_special_tokens=True).strip()
        return text, int(inputs["input_ids"].shape[1]), int(output.shape[1])


def judge_correctness(
    judge: LocalSeq2Seq,
    *,
    question: str,
    reference: str,
    candidate: str,
    evidence: str,
) -> int | None:
    prompt = (
        "Score candidate correctness 0 to 4 using the reference and evidence. "
        "4 fully correct; 3 mostly correct; 2 partially correct; "
        "1 mostly incorrect with a small correct element; 0 incorrect or unsupported. "
        "Return only one digit.\n\n"
        f"QUESTION:\n{question}\nREFERENCE:\n{reference}\n"
        f"EVIDENCE:\n{evidence[:5000]}\nCANDIDATE:\n{candidate}"
    )
    raw, _, _ = judge.generate(prompt, max_new_tokens=8)
    match = re.search(r"\b([0-4])\b", raw)
    return int(match.group(1)) if match else None


class NLISupport:
    def __init__(self, model_id: str = NLI_ID) -> None:
        self.model = CrossEncoder(model_id)
        labels = {
            int(index): str(label).lower()
            for index, label in self.model.model.config.id2label.items()
        }
        self.entail_index = next(
            (index for index, label in labels.items() if "entail" in label),
            1,
        )

    def probabilities(self, evidence_claim_pairs: list[tuple[str, str]]) -> list[float]:
        if not evidence_claim_pairs:
            return []
        logits = np.asarray(self.model.predict(evidence_claim_pairs))
        if logits.ndim == 1:
            return [float(x) for x in logits]
        probabilities = softmax(logits, axis=1)
        return [float(row[self.entail_index]) for row in probabilities]


def support_metrics(
    answer: str,
    hits,
    nli: NLISupport,
    *,
    threshold: float = 0.5,
) -> tuple[float | None, float, float]:
    claims = split_claims(answer)
    citations = sorted(set(parse_citations(answer)))
    valid_citations = [i for i in citations if 1 <= i <= len(hits)]
    if not claims:
        return None, 0.0, 0.0
    if not valid_citations:
        return 0.0, 0.0, 0.0

    pairs: list[tuple[str, str]] = []
    coordinates: list[tuple[int, int]] = []
    for claim_i, claim in enumerate(claims):
        for citation in valid_citations:
            pairs.append((hits[citation - 1].chunk.text, claim))
            coordinates.append((claim_i, citation))
    probabilities = nli.probabilities(pairs)

    supported_claims: set[int] = set()
    supported_citations: set[int] = set()
    for (claim_i, citation), probability in zip(coordinates, probabilities):
        if probability >= threshold:
            supported_claims.add(claim_i)
            supported_citations.add(citation)

    faithfulness = len(supported_claims) / len(claims)
    citation_precision = len(supported_citations) / len(valid_citations)
    citation_recall = len(supported_claims) / len(claims)
    return faithfulness, citation_precision, citation_recall


def mean_ci(values: list[float]) -> dict[str, object]:
    if not values:
        return {"mean": None, "ci95": None, "n": 0}
    lo, hi = bootstrap_ci(values, samples=2000, seed=17)
    return {"mean": float(np.mean(values)), "ci95": [lo, hi], "n": len(values)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("chunks", type=Path)
    parser.add_argument("human_dataset", type=Path)
    parser.add_argument(
        "--synthetic-baseline",
        type=Path,
        default=Path("results/bootstrap/retrieval_baseline.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/final/metrics.json"),
    )
    parser.add_argument(
        "--failures",
        type=Path,
        default=Path("results/final/failure_cases.json"),
    )
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    args = parser.parse_args()

    chunks = load_chunks(args.chunks)
    by_id = {chunk.id: chunk for chunk in chunks}
    items = load_items(args.human_dataset)
    if len(items) < 100:
        raise ValueError(f"final human evaluation requires >=100 items; found {len(items)}")

    embedder = SentenceTransformerEmbedder(args.embedding_model)
    vectors = embedder.encode([chunk.text for chunk in chunks])
    store = InMemoryVectorStore(args.embedding_model)
    store.add(chunks, vectors)
    bm25 = BM25Index(chunks)

    generator = LocalSeq2Seq(GENERATOR_ID)
    judge = LocalSeq2Seq(JUDGE_ID)
    nli = NLISupport()

    rows: list[dict] = []
    failures: list[dict] = []
    predicted_abstain: list[bool] = []
    gold_unanswerable: list[bool] = []

    for item in items:
        retrieval_started = perf_counter()
        dense = dense_search(item["question"], embedder=embedder, store=store, k=20)
        sparse = sparse_search(item["question"], index=bm25, k=20)
        hits = hybrid_rrf(dense, sparse, k=20)
        retrieval_ms = (perf_counter() - retrieval_started) * 1000.0

        gold_sources = set(item.get("gold_sources") or [])
        if not gold_sources:
            gold_sources = {
                by_id[cid].source for cid in item.get("gold_chunk_ids", []) if cid in by_id
            }

        retrieval_row = {}
        retrieval_recall5 = None
        if item["answerable"] and gold_sources:
            ranked_sources = unique([hit.chunk.source for hit in hits])
            metric = retrieval_metrics(ranked_sources, gold_sources, ks=KS)
            retrieval_recall5 = metric.recall_at_k[5]
            retrieval_row = {
                "mrr": metric.mrr,
                **{f"recall@{k}": metric.recall_at_k[k] for k in KS},
                **{f"ndcg@{k}": metric.ndcg_at_k[k] for k in KS},
            }

        final_hits = hits[:5]
        prompt = build_prompt(item["question"], final_hits, max_chars=16000)
        generation_started = perf_counter()
        answer, prompt_tokens, completion_tokens = generator.generate(prompt, max_new_tokens=128)
        generation_ms = (perf_counter() - generation_started) * 1000.0

        abstained = ABSTENTION in answer.lower()
        predicted_abstain.append(abstained)
        gold_unanswerable.append(not item["answerable"])

        citation_validation = validate_citations(answer, len(final_hits))
        faithfulness, citation_precision, citation_recall = support_metrics(
            answer,
            final_hits,
            nli,
        )

        evidence = "\n\n".join(hit.chunk.text for hit in final_hits)
        correctness = None
        judge_error = False
        if item["answerable"]:
            correctness = judge_correctness(
                judge,
                question=item["question"],
                reference=item.get("gold_answer") or "",
                candidate=answer,
                evidence=evidence,
            )
            judge_error = correctness is None
        else:
            correctness = 4 if abstained else 0

        exact_match = None
        if item["question_type"] in {"numeric", "factoid"}:
            gold = item.get("gold_answer") or ""
            if item["question_type"] == "numeric":
                expected = numeric_values(gold)
                actual = numeric_values(answer)
                exact_match = float(bool(expected) and expected.issubset(actual))
            else:
                expected = normalize(gold)
                actual = normalize(answer)
                exact_match = float(bool(expected) and expected in actual)

        row = {
            "id": item["id"],
            "question_type": item["question_type"],
            "answerable": item["answerable"],
            "answer": answer,
            "abstained": abstained,
            "citations_valid": citation_validation.valid,
            "invalid_citations": citation_validation.invalid,
            "faithfulness": faithfulness,
            "citation_precision": citation_precision,
            "citation_recall": citation_recall,
            "correctness_0_4": correctness,
            "exact_match": exact_match,
            "retrieval_ms": retrieval_ms,
            "generation_ms": generation_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": 0.0,
            **retrieval_row,
        }
        rows.append(row)

        categories = []
        if item["answerable"] and retrieval_recall5 == 0.0:
            categories.append("retrieval_miss")
        if item["answerable"] and abstained:
            categories.append("wrong_abstention")
        if not item["answerable"] and not abstained:
            categories.append("wrong_abstention")
        if faithfulness is not None and faithfulness < 1.0:
            categories.append("unsupported_claim")
        if not citation_validation.valid or citation_recall < 1.0:
            categories.append("citation_failure")
        if judge_error:
            categories.append("judge_error")
        if categories:
            failures.append(
                {
                    "id": item["id"],
                    "categories": categories,
                    "answer": answer,
                    "gold_answer": item.get("gold_answer"),
                    "retrieved_sources": [hit.chunk.source for hit in final_hits],
                }
            )

    answerable_rows = [row for row in rows if row["answerable"]]
    correctness_values = [
        row["correctness_0_4"] / 4.0
        for row in answerable_rows
        if row["correctness_0_4"] is not None
    ]
    faithfulness_values = [
        row["faithfulness"] for row in rows if row["faithfulness"] is not None
    ]
    citation_precision_values = [row["citation_precision"] for row in rows]
    citation_recall_values = [row["citation_recall"] for row in rows]
    retrieval_mrr_values = [row["mrr"] for row in answerable_rows if "mrr" in row]
    retrieval_recall5_values = [
        row["recall@5"] for row in answerable_rows if "recall@5" in row
    ]
    exact_values = [row["exact_match"] for row in rows if row["exact_match"] is not None]
    abstention = abstention_metrics(predicted_abstain, gold_unanswerable)

    human_summary = {
        "question_count": len(rows),
        "answerable_count": sum(row["answerable"] for row in rows),
        "unanswerable_count": sum(not row["answerable"] for row in rows),
        "retrieval": {
            "mrr": mean_ci(retrieval_mrr_values),
            "recall@5": mean_ci(retrieval_recall5_values),
        },
        "generation": {
            "correctness": mean_ci(correctness_values),
            "faithfulness": mean_ci(faithfulness_values),
            "citation_precision": mean_ci(citation_precision_values),
            "citation_recall": mean_ci(citation_recall_values),
            "exact_numeric_factoid_match": mean_ci(exact_values),
            **abstention,
        },
        "system": {
            "retrieval_ms": mean_ci([row["retrieval_ms"] for row in rows]),
            "generation_ms": mean_ci([row["generation_ms"] for row in rows]),
            "prompt_tokens_total": sum(row["prompt_tokens"] for row in rows),
            "completion_tokens_total": sum(row["completion_tokens"] for row in rows),
            "cost_usd_total": 0.0,
        },
    }

    synthetic = None
    if args.synthetic_baseline.exists():
        synthetic = json.loads(args.synthetic_baseline.read_text(encoding="utf-8"))

    result = {
        "models": {
            "embedding": args.embedding_model,
            "generator": GENERATOR_ID,
            "correctness_judge": JUDGE_ID,
            "faithfulness_nli": NLI_ID,
        },
        "human_verified": human_summary,
        "synthetic_retrieval": synthetic,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    args.failures.parent.mkdir(parents=True, exist_ok=True)
    args.failures.write_text(json.dumps(failures, indent=2), encoding="utf-8")
    print(json.dumps({"human_verified": human_summary, "failure_count": len(failures)}, indent=2))


if __name__ == "__main__":
    main()
