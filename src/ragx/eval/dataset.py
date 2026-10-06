from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class EvalItem(BaseModel):
    id: str
    question: str
    question_type: Literal[
        "single_hop", "multi_hop", "numeric", "factoid", "list", "unanswerable"
    ]
    answerable: bool
    split: Literal["synthetic", "human_verified"]
    gold_chunk_ids: list[str] = Field(default_factory=list)
    gold_answer: str | None = None
    ambiguous: bool = False
    notes: str | None = None


def load_jsonl(path: str | Path) -> list[EvalItem]:
    items: list[EvalItem] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            items.append(EvalItem.model_validate_json(line))
    return items


def dataset_checksum(items: list[EvalItem]) -> str:
    canonical = "\n".join(
        json.dumps(item.model_dump(), sort_keys=True, separators=(",", ":"))
        for item in sorted(items, key=lambda x: x.id)
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def validate_human_subset(items: list[EvalItem]) -> None:
    verified = [x for x in items if x.split == "human_verified"]
    if len(verified) < 100:
        raise ValueError(f"human-verified subset requires >=100 items; found {len(verified)}")
    missing = [x.id for x in verified if not x.gold_answer or (x.answerable and not x.gold_chunk_ids)]
    if missing:
        raise ValueError(f"incomplete human-verified gold items: {missing[:10]}")
