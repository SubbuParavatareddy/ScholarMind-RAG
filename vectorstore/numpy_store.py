from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class NumpyVectorStore:
    """In-memory cosine-similarity store backed by NumPy.

    Supports MMR (Maximal Marginal Relevance) retrieval.
    No SQLite, no protobuf, no external services.
    """

    texts: list = field(default_factory=list)
    metadatas: list = field(default_factory=list)
    embeddings: np.ndarray = field(default_factory=lambda: np.empty((0, 384)))

    def add(self, texts: list[str], metadatas: list[dict], embeddings: list[list[float]]):
        self.texts.extend(texts)
        self.metadatas.extend(metadatas)
        arr = np.array(embeddings, dtype=np.float32)
        if self.embeddings.shape[0] == 0:
            self.embeddings = arr
        else:
            self.embeddings = np.vstack([self.embeddings, arr])

    def _cosine(self, query_vec: np.ndarray) -> np.ndarray:
        if self.embeddings.shape[0] == 0:
            return np.array([])
        q = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-10
        normed = self.embeddings / norms
        return (normed @ q).ravel()

    def similarity_search(self, query_vec: np.ndarray, k: int = 4) -> list[dict]:
        scores = self._cosine(query_vec)
        if len(scores) == 0:
            return []
        idx = np.argsort(scores)[::-1][:k]
        return [{"text": self.texts[i], "metadata": self.metadatas[i], "score": float(scores[i])}
                for i in idx]

    def mmr_search(self, query_vec: np.ndarray, k: int = 4, fetch_k: int = 20,
                   lambda_mult: float = 0.6) -> list[dict]:
        scores = self._cosine(query_vec)
        if len(scores) == 0:
            return []
        fetch_k = min(fetch_k, len(scores))
        cands = np.argsort(scores)[::-1][:fetch_k]
        selected: list[int] = []
        while len(selected) < k and len(cands) > 0:
            if not selected:
                best = cands[0]
            else:
                sel_embs = self.embeddings[selected]
                cand_embs = self.embeddings[cands]
                sel_n = sel_embs / (np.linalg.norm(sel_embs, axis=1, keepdims=True) + 1e-10)
                can_n = cand_embs / (np.linalg.norm(cand_embs, axis=1, keepdims=True) + 1e-10)
                sim_to_sel = (can_n @ sel_n.T).max(axis=1)
                rel = scores[cands]
                mmr_scores = lambda_mult * rel - (1 - lambda_mult) * sim_to_sel
                best = cands[np.argmax(mmr_scores)]
            selected.append(best)
            cands = cands[cands != best]
        return [{"text": self.texts[i], "metadata": self.metadatas[i], "score": float(scores[i])}
                for i in selected]

    def get_all_texts(self) -> list[str]:
        return list(self.texts)
