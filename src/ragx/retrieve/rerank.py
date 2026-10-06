from __future__ import annotations

from sentence_transformers import CrossEncoder

from ragx.index.base import SearchHit


class CrossEncoderReranker:
    def __init__(self, model_id: str = "BAAI/bge-reranker-base") -> None:
        self.model_id = model_id
        self.model = CrossEncoder(model_id)

    def rerank(self, query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]:
        if not hits or top_k <= 0:
            return []
        pairs = [(query, h.chunk.text) for h in hits]
        scores = self.model.predict(pairs)
        ranked = sorted(
            zip(hits, scores),
            key=lambda item: (-float(item[1]), item[0].chunk.id),
        )[:top_k]
        return [
            SearchHit(chunk=hit.chunk, score=float(score), rank=rank + 1)
            for rank, (hit, score) in enumerate(ranked)
        ]
