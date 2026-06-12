from __future__ import annotations
import re
import json

from vectorstore.numpy_store import NumpyVectorStore
from retrieval.rag_engine import RAGEngine
from prompts.templates import (
    SUMMARY_PROMPT,
    COMPARE_PROMPT,
    FLASHCARD_PROMPT,
    GAP_PROMPT,
    ELI15_PROMPT,
)


class PaperAnalyzer:
    """Generates AI-driven analysis for research papers.

    The cache parameter is a dict-like object (e.g. st.session_state or a plain dict
    in tests) that persists results between Streamlit reruns.
    """

    def __init__(self, rag_engine: RAGEngine, cache: dict):
        self._rag = rag_engine
        self._cache = cache

    # ── private helpers ───────────────────────────────────────────────────────

    def _llm_json(self, prompt: str, temp: float = 0.1):
        raw = self._rag.llm(temp).invoke(prompt)
        text = raw.content if hasattr(raw, "content") else str(raw)
        text = re.sub(r"```json|```", "", text).strip()
        return json.loads(text)

    @staticmethod
    def _sample(vs: NumpyVectorStore, n: int = 12) -> str:
        texts = vs.get_all_texts()
        mid = len(texts) // 2
        sample = texts[:4] + texts[mid:mid + 4] + texts[-4:]
        return "\n\n---\n\n".join(sample[:n])[:6000]

    def _ensure(self, key: str, default):
        if key not in self._cache:
            self._cache[key] = default
        return self._cache[key]

    # ── public API ────────────────────────────────────────────────────────────

    def get_summary(self, col_id: str, vs: NumpyVectorStore) -> dict:
        summaries = self._ensure("summaries", {})
        if col_id not in summaries:
            try:
                data = self._llm_json(SUMMARY_PROMPT.format(ctx=self._sample(vs)))
                summaries[col_id] = data if isinstance(data, dict) else {}
            except Exception as e:
                summaries[col_id] = {"one_liner": f"Summary error: {e}"}
        return summaries[col_id]

    def get_flashcards(self, col_id: str, vs: NumpyVectorStore) -> list:
        key = f"fc_{col_id}"
        if key not in self._cache:
            try:
                data = self._llm_json(FLASHCARD_PROMPT.format(ctx=self._sample(vs)), temp=0.3)
                self._cache[key] = data if isinstance(data, list) else []
            except Exception:
                self._cache[key] = []
        return self._cache[key]

    def get_gaps(self, col_id: str, vs: NumpyVectorStore) -> dict:
        key = f"gaps_{col_id}"
        if key not in self._cache:
            texts = vs.get_all_texts()
            mid = len(texts) // 2
            ctx = "\n\n".join(texts[mid:] + texts[-4:])[:5500]
            try:
                data = self._llm_json(GAP_PROMPT.format(ctx=ctx))
                self._cache[key] = data if isinstance(data, dict) else {}
            except Exception:
                self._cache[key] = {}
        return self._cache[key]

    def get_eli15(self, col_id: str, vs: NumpyVectorStore) -> str:
        key = f"eli15_{col_id}"
        if key not in self._cache:
            ctx = "\n\n".join(vs.get_all_texts()[:6])[:4000]
            raw = self._rag.llm(temp=0.5).invoke(ELI15_PROMPT.format(ctx=ctx))
            self._cache[key] = raw.content if hasattr(raw, "content") else str(raw)
        return self._cache[key]

    def get_compare(self, id_a: str, id_b: str, collections: dict) -> dict:
        compare_cache = self._ensure("compare_cache", {})
        key = f"{id_a}__{id_b}"
        rkey = f"{id_b}__{id_a}"
        if key not in compare_cache and rkey not in compare_cache:
            ma = collections[id_a]
            mb = collections[id_b]
            ca = "\n\n".join(ma["vectorstore"].get_all_texts()[:8])[:3000]
            cb = "\n\n".join(mb["vectorstore"].get_all_texts()[:8])[:3000]
            try:
                data = self._llm_json(COMPARE_PROMPT.format(
                    na=ma["filename"], ca=ca, nb=mb["filename"], cb=cb))
                compare_cache[key] = data if isinstance(data, dict) else {}
            except Exception:
                compare_cache[key] = {}
        return compare_cache.get(key) or compare_cache.get(rkey, {})
