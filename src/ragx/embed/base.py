from __future__ import annotations

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    model_id: str
    max_sequence_length: int

    def encode(self, texts: list[str]) -> np.ndarray: ...

    def tokenize(self, text: str) -> list[str]: ...
