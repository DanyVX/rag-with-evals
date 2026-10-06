from __future__ import annotations

import re
from typing import Protocol, TypeVar

from transformers import AutoTokenizer

Token = TypeVar("Token")


class TokenCodec(Protocol):
    max_sequence_length: int

    def encode(self, text: str) -> list: ...

    def decode(self, tokens: list) -> str: ...


class WhitespaceTokenCodec:
    def __init__(self, max_sequence_length: int = 100_000) -> None:
        self.max_sequence_length = max_sequence_length

    def encode(self, text: str) -> list[str]:
        return re.findall(r"\S+", text)

    def decode(self, tokens: list[str]) -> str:
        return " ".join(tokens)


class HuggingFaceTokenCodec:
    def __init__(self, model_id: str, *, max_sequence_length: int | None = None) -> None:
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        reported = int(getattr(self.tokenizer, "model_max_length", 0) or 0)
        if reported <= 0 or reported > 1_000_000:
            reported = max_sequence_length or 512
        self.max_sequence_length = min(reported, max_sequence_length or reported)

    def encode(self, text: str) -> list[int]:
        return list(
            self.tokenizer.encode(
                text,
                add_special_tokens=False,
                truncation=False,
            )
        )

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(
            tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
