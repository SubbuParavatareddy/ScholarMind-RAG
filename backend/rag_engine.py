from __future__ import annotations
import re
import json
import numpy as np

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .vector_store import NumpyVectorStore

_RAG_TEMPLATE = """You are ScholarMind, a precise AI research assistant.
Answer using ONLY the retrieved context. Be structured and cite evidence.
If context is insufficient, say so — never fabricate.

Context:
{context}

Question: {question}

Answer:"""

_FOLLOW_UP_TEMPLATE = """Suggest 3 follow-up questions. Return ONLY JSON array of 3 strings.
Q: {q}
A: {a}
JSON:"""

_HEDGES = {"don't have", "not enough", "cannot", "no information", "not provided"}


class RAGEngine:
    """Handles embedding-based retrieval and LLM-driven RAG responses."""

    def __init__(self, api_key: str, embedding_model):
        self._api_key = api_key
        self._emb = embedding_model
        self._rag_prompt = ChatPromptTemplate.from_template(_RAG_TEMPLATE)

    def llm(self, temp: float = 0.15) -> ChatGoogleGenerativeAI:
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=self._api_key,
            temperature=temp,
        )

    def query(self, question: str, vs: NumpyVectorStore, top_k: int) -> dict:
        q_vec = np.array(self._emb.embed_query(question), dtype=np.float32)
        results = vs.mmr_search(q_vec, k=top_k, fetch_k=min(top_k * 3, len(vs.texts)))
        context = "\n\n---\n\n".join(r["text"] for r in results)

        chain = self._rag_prompt | self.llm() | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})

        seen: set[str] = set()
        sources: list[dict] = []
        for r in results:
            txt = r["text"].strip()
            if txt in seen:
                continue
            seen.add(txt)
            sources.append({
                "content": txt[:450] + ("…" if len(txt) > 450 else ""),
                "page": r["metadata"].get("page", "–"),
            })

        conf = ("low" if any(h in answer.lower() for h in _HEDGES)
                else "medium" if len(answer.strip()) < 130 else "high")
        return {"answer": answer, "sources": sources, "confidence": conf}

    def get_follow_ups(self, question: str, answer: str) -> list[str]:
        try:
            raw = self.llm(temp=0.4).invoke(
                _FOLLOW_UP_TEMPLATE.format(q=question, a=answer[:600])
            )
            text = raw.content if hasattr(raw, "content") else str(raw)
            text = re.sub(r"```json|```", "", text).strip()
            data = json.loads(text)
            return data[:3] if isinstance(data, list) else []
        except Exception:
            return []
