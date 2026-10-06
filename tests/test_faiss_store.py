from pathlib import Path

import numpy as np
import pytest

from ragx.chunk.core import Chunk
from ragx.index.faiss_sqlite import FaissSQLiteStore


def chunk(cid: str, text: str) -> Chunk:
    return Chunk(
        id=cid,
        doc_hash="doc",
        source="source.txt",
        text=text,
        position=0,
    )


def test_store_matches_indexed_chunks(tmp_path: Path) -> None:
    chunks = [chunk("a", "alpha"), chunk("b", "beta")]
    store = FaissSQLiteStore(tmp_path / "index", "model-a")
    store.add(chunks, np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    assert store.matches_chunks(chunks)
    assert not store.matches_chunks([chunks[0]])
    assert not store.matches_chunks([chunk("a", "alpha"), chunk("c", "gamma")])


def test_embedding_model_change_is_refused(tmp_path: Path) -> None:
    store = FaissSQLiteStore(tmp_path / "index", "model-a")
    store.add([chunk("a", "alpha")], np.asarray([[1.0, 0.0]], dtype=np.float32))
    with pytest.raises(ValueError, match="embedding model changed"):
        FaissSQLiteStore(tmp_path / "index", "model-b")
