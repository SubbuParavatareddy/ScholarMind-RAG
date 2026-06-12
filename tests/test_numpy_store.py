"""Unit tests for NumpyVectorStore — no Streamlit, no LLM, no API key needed."""
import numpy as np
import pytest

from vectorstore.numpy_store import NumpyVectorStore


@pytest.fixture
def store():
    vs = NumpyVectorStore()
    texts = ["The transformer uses self-attention.", "BERT is a bidirectional model.", "GPT is autoregressive."]
    metas = [{"page": i} for i in range(3)]
    # 384-dim random unit vectors
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((3, 384)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    vs.add(texts, metas, vecs.tolist())
    return vs, vecs


def test_add_and_get_all_texts(store):
    vs, _ = store
    assert len(vs.get_all_texts()) == 3


def test_similarity_search_returns_k(store):
    vs, vecs = store
    results = vs.similarity_search(vecs[0], k=2)
    assert len(results) == 2


def test_similarity_search_top_is_self(store):
    vs, vecs = store
    results = vs.similarity_search(vecs[0], k=1)
    assert results[0]["text"] == "The transformer uses self-attention."


def test_mmr_search_no_duplicates(store):
    vs, vecs = store
    results = vs.mmr_search(vecs[0], k=3, fetch_k=3)
    texts = [r["text"] for r in results]
    assert len(texts) == len(set(texts))


def test_empty_store_returns_empty():
    vs = NumpyVectorStore()
    q = np.zeros(384, dtype=np.float32)
    assert vs.similarity_search(q) == []
    assert vs.mmr_search(q) == []
