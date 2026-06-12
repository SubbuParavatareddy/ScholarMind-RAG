"""Unit tests for DocumentLoader — uses a mock embedding model, no API key needed."""
import io
import pytest

from ingestion.document_loader import DocumentLoader
from vectorstore.numpy_store import NumpyVectorStore


class _FakeEmbedder:
    """Returns deterministic 384-dim vectors for any text list."""

    def embed_documents(self, texts):
        import numpy as np
        rng = np.random.default_rng(0)
        return rng.standard_normal((len(texts), 384)).tolist()


def _txt_file(content: str, name: str = "paper.txt"):
    buf = io.BytesIO(content.encode())
    buf.name = name
    buf.read = buf.read  # already has read()
    return buf


def test_ingest_txt_returns_collection():
    text = ("This is a test paper. " * 60)  # enough chars for at least one chunk
    f = _txt_file(text)
    result = DocumentLoader.ingest(f, _FakeEmbedder())
    assert result["filename"] == "paper.txt"
    assert result["chunks"] >= 1
    assert isinstance(result["vectorstore"], NumpyVectorStore)
    assert len(result["col_id"]) == 8


def test_ingest_unsupported_extension_raises():
    f = _txt_file("content", name="paper.docx")
    with pytest.raises(ValueError, match="Unsupported type"):
        DocumentLoader.ingest(f, _FakeEmbedder())


def test_chunk_params():
    assert DocumentLoader.CHUNK_SIZE == 900
    assert DocumentLoader.CHUNK_OVERLAP == 180
