from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbedder:
    def __init__(self, model_id: str = "BAAI/bge-small-en-v1.5") -> None:
        self.model_id = model_id
        self.model = SentenceTransformer(model_id)
        self.max_sequence_length = int(self.model.max_seq_length)

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32)

    def tokenize(self, text: str) -> list[str]:
        tokenizer = self.model.tokenizer
        ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
        return [str(i) for i in ids]
