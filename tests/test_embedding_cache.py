import numpy as np

from ragx.embed.cache import CachedEmbedder, EmbeddingCache, embedding_cache_key


class FakeEmbedder:
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.max_sequence_length = 16
        self.calls = 0

    def encode(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        return np.asarray(
            [[float(len(text)), float(index + 1)] for index, text in enumerate(texts)],
            dtype=np.float32,
        )

    def tokenize(self, text: str) -> list[str]:
        return text.split()


def test_cache_reuses_vectors_for_same_model_and_preprocess(tmp_path) -> None:
    base = FakeEmbedder("model-a")
    cached = CachedEmbedder(base, EmbeddingCache(tmp_path / "emb.sqlite3"), preprocess_version="v1")
    first = cached.encode(["alpha", "beta"])
    second = cached.encode(["alpha", "beta"])
    assert base.calls == 1
    assert np.array_equal(first, second)


def test_preprocess_version_invalidates_cache(tmp_path) -> None:
    cache = EmbeddingCache(tmp_path / "emb.sqlite3")
    first_base = FakeEmbedder("model-a")
    second_base = FakeEmbedder("model-a")
    CachedEmbedder(first_base, cache, preprocess_version="v1").encode(["alpha"])
    CachedEmbedder(second_base, cache, preprocess_version="v2").encode(["alpha"])
    assert first_base.calls == 1
    assert second_base.calls == 1
    assert cache.count() == 2


def test_model_id_invalidates_cache(tmp_path) -> None:
    cache = EmbeddingCache(tmp_path / "emb.sqlite3")
    CachedEmbedder(FakeEmbedder("model-a"), cache, preprocess_version="v1").encode(["alpha"])
    other = FakeEmbedder("model-b")
    CachedEmbedder(other, cache, preprocess_version="v1").encode(["alpha"])
    assert other.calls == 1
    assert cache.count() == 2


def test_cache_key_includes_model_and_preprocess() -> None:
    a = embedding_cache_key("m1", "v1", "text")
    b = embedding_cache_key("m2", "v1", "text")
    c = embedding_cache_key("m1", "v2", "text")
    assert len({a, b, c}) == 3
