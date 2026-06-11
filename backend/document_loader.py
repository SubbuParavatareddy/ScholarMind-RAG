from __future__ import annotations
import os
import hashlib
import tempfile
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .vector_store import NumpyVectorStore


class DocumentLoader:
    """Loads, splits, embeds, and indexes research papers into a NumpyVectorStore."""

    CHUNK_SIZE = 900
    CHUNK_OVERLAP = 180
    SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", " "]
    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}

    @classmethod
    def ingest(cls, uploaded_file, embedding_model) -> dict:
        """Process an uploaded file and return a collection dict with vectorstore and stats.

        Raises ValueError for unsupported types or empty documents.
        """
        ext = Path(uploaded_file.name).suffix.lower()
        if ext not in cls.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported type '{ext}'. Use PDF, TXT or MD.")

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(uploaded_file.read())
            path = tmp.name
        try:
            docs = (PyPDFLoader(path).load() if ext == ".pdf"
                    else TextLoader(path, encoding="utf-8").load())
        finally:
            os.unlink(path)

        if not docs:
            raise ValueError("No text extracted — ensure the PDF contains selectable text.")

        chunks = RecursiveCharacterTextSplitter(
            chunk_size=cls.CHUNK_SIZE,
            chunk_overlap=cls.CHUNK_OVERLAP,
            separators=cls.SEPARATORS,
        ).split_documents(docs)

        if not chunks:
            raise ValueError("Document produced no usable chunks.")

        texts = [c.page_content for c in chunks]
        metas = [c.metadata for c in chunks]
        vectors = embedding_model.embed_documents(texts)

        vs = NumpyVectorStore()
        vs.add(texts, metas, vectors)

        col_id = hashlib.md5((uploaded_file.name + str(len(chunks))).encode()).hexdigest()[:8]
        return {
            "filename": uploaded_file.name,
            "chunks": len(chunks),
            "vectorstore": vs,
            "col_id": col_id,
            "pages": len(docs),
        }
