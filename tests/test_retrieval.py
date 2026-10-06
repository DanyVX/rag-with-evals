import numpy as np

from ragx.chunk.core import Chunk
from ragx.index.base import SearchHit
from ragx.retrieve.advanced import apply_score_floor, mmr_select
from ragx.retrieve.core import hybrid_rrf


def _hit(cid: str, score: float, rank: int) -> SearchHit:
    return SearchHit(
        chunk=Chunk(id=cid, doc_hash="d", source="x", text=cid, position=rank),
        score=score,
        rank=rank,
    )


def test_rrf_is_deterministic() -> None:
    dense = [_hit("a", 0.9, 1), _hit("b", 0.8, 2)]
    sparse = [_hit("b", 4.0, 1), _hit("a", 3.0, 2)]
    one = hybrid_rrf(dense, sparse, k=2)
    two = hybrid_rrf(dense, sparse, k=2)
    assert [x.chunk.id for x in one] == [x.chunk.id for x in two]


def test_score_floor() -> None:
    assert [h.chunk.id for h in apply_score_floor([_hit("a", 0.2, 1), _hit("b", 0.8, 2)], 0.5)] == ["b"]


def test_mmr_returns_requested_count() -> None:
    hits = [_hit("a", 1.0, 1), _hit("b", 0.9, 2), _hit("c", 0.8, 3)]
    vectors = np.asarray([[1, 0], [0.9, 0.1], [0, 1]], dtype=np.float32)
    out = mmr_select(np.asarray([1, 0], dtype=np.float32), vectors, hits, k=2)
    assert len(out) == 2
