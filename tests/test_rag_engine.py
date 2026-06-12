"""Unit tests for RAGEngine — uses mock LLM and embedder, no real API calls."""
import numpy as np
import pytest

from vectorstore.numpy_store import NumpyVectorStore
from retrieval.rag_engine import RAGEngine


class _FakeEmbedder:
    def embed_query(self, text):
        rng = np.random.default_rng(len(text))
        v = rng.standard_normal(384).astype(np.float32)
        return (v / np.linalg.norm(v)).tolist()


class _FakeLLM:
    """Mimics ChatGoogleGenerativeAI.invoke() return shape."""

    def __init__(self, response: str):
        self._response = response

    def invoke(self, *args, **kwargs):
        class _Resp:
            content = None
        r = _Resp()
        r.content = self._response
        return r


def _make_engine(llm_response="Test answer about attention."):
    engine = RAGEngine.__new__(RAGEngine)
    engine._api_key = "fake"
    engine._emb = _FakeEmbedder()

    from langchain_core.prompts import ChatPromptTemplate
    from prompts.templates import RAG_TEMPLATE
    engine._rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
    engine._fake_llm = _FakeLLM(llm_response)
    # Patch llm() to return a mock that supports LCEL pipe operator
    return engine


def _make_store() -> NumpyVectorStore:
    vs = NumpyVectorStore()
    rng = np.random.default_rng(7)
    vecs = rng.standard_normal((5, 384)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    vs.add(
        ["Attention is all you need.", "Self-attention mechanism.", "Multi-head attention.",
         "Encoder decoder architecture.", "Positional encoding."],
        [{"page": i} for i in range(5)],
        vecs.tolist(),
    )
    return vs


def test_confidence_high():
    from retrieval.rag_engine import RAGEngine
    from prompts.templates import HEDGE_PHRASES
    answer = "The paper proposes a transformer model based on self-attention with 8 heads and positional encoding that achieves state-of-the-art results on translation benchmarks."
    assert not any(h in answer.lower() for h in HEDGE_PHRASES)
    assert len(answer.strip()) >= 130


def test_confidence_low():
    from prompts.templates import HEDGE_PHRASES
    answer = "I cannot find this in the provided context."
    assert any(h in answer.lower() for h in HEDGE_PHRASES)


def test_get_follow_ups_returns_list_on_bad_json():
    engine = _make_engine()

    class _BadLLM:
        def invoke(self, *a, **kw):
            class R:
                content = "not json at all"
            return R()

    import unittest.mock as mock
    with mock.patch.object(engine, "llm", return_value=_BadLLM()):
        result = engine.get_follow_ups("q", "a")
    assert isinstance(result, list)
