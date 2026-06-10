"""
ScholarMind — AI Research Paper Assistant
Gemini 2.0 Flash · LangChain 0.3.x LCEL · ChromaDB 0.6.x · Streamlit 1.41

KEY FIX: Uses LCEL (LangChain Expression Language) instead of legacy RetrievalQA.
RetrievalQA inherits from Chain which uses pydantic forward-refs that break on
Python 3.14. LCEL is fully Python 3.12/3.14 safe.

Requires .python-version = 3.12  (in repo root, respected by Streamlit Cloud)
"""

import os, re, json, hashlib, tempfile, datetime
import streamlit as st
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

import plotly.graph_objects as go

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ScholarMind · Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg0: #05080F; --bg1: #0C1220; --bg2: #111C2E; --bg3: #172035; --bg4: #1D2840;
  --border: rgba(99,120,180,0.13); --border-s: rgba(99,120,180,0.25);
  --accent: #7C83FD; --accent-d: #5B5FD9; --accent-glow: rgba(124,131,253,0.18);
  --green: #34D399; --amber: #FBBF24; --red: #F87171;
  --text0: #EEF0F8; --text1: #A0ABCC; --text2: #5A6A8A; --text3: #2D3A52;
  --font-sans: 'Inter', system-ui, sans-serif;
  --font-mono: 'JetBrains Mono', monospace;
  --r-sm: 6px; --r-md: 10px; --r-lg: 14px;
}

*, *::before, *::after { box-sizing: border-box; }

[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main .block-container { background: var(--bg0) !important; }

[data-testid="block-container"] {
  padding: 1.25rem 2rem 3rem !important;
  max-width: 1100px;
}

[data-testid="stSidebar"] {
  background: var(--bg1) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebarContent"] { padding: 0 12px 24px !important; }
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] small { color: var(--text1) !important; font-size: 12px !important; }

[data-testid="stTextInput"] input {
  background: var(--bg3) !important; border: 1px solid var(--border) !important;
  border-radius: var(--r-md) !important; color: var(--text0) !important;
  font-family: var(--font-mono) !important; font-size: 12px !important;
}
[data-testid="stTextInput"] input:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px var(--accent-glow) !important; outline: none !important;
}

.stButton > button {
  background: var(--bg3) !important; border: 1px solid var(--border) !important;
  color: var(--text1) !important; border-radius: var(--r-md) !important;
  font-family: var(--font-sans) !important; font-size: 13px !important;
  font-weight: 500 !important; transition: all 0.15s ease !important;
}
.stButton > button:hover {
  border-color: var(--accent) !important; color: var(--text0) !important;
  background: var(--accent-glow) !important;
}
.stButton > button[kind="primary"] {
  background: var(--accent) !important; border-color: var(--accent) !important; color: #fff !important;
}
.stButton > button[kind="primary"]:hover {
  background: var(--accent-d) !important; border-color: var(--accent-d) !important;
}

[data-testid="stTabs"] [role="tablist"] {
  background: var(--bg1) !important; border-bottom: 1px solid var(--border) !important;
  gap: 2px !important; padding: 6px 8px 0 !important;
  border-radius: var(--r-lg) var(--r-lg) 0 0;
}
[data-testid="stTabs"] button[role="tab"] {
  background: transparent !important; border: none !important;
  color: var(--text2) !important; font-family: var(--font-sans) !important;
  font-size: 12.5px !important; font-weight: 500 !important;
  padding: 7px 15px !important; border-radius: 8px 8px 0 0 !important;
  transition: all 0.15s !important;
}
[data-testid="stTabs"] button[role="tab"]:hover { color: var(--text0) !important; background: var(--bg3) !important; }
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
  color: var(--accent) !important; background: var(--bg2) !important;
  border-bottom: 2px solid var(--accent) !important;
}
[data-testid="stTabsContent"] {
  background: var(--bg2) !important; border: 1px solid var(--border) !important;
  border-top: none !important; border-radius: 0 0 var(--r-lg) var(--r-lg) !important;
  padding: 24px !important; min-height: 520px !important;
}

[data-testid="stExpander"] {
  background: var(--bg3) !important; border: 1px solid var(--border) !important;
  border-radius: var(--r-md) !important; margin-bottom: 8px !important;
}
[data-testid="stExpander"] summary {
  color: var(--text1) !important; font-size: 13px !important;
  font-family: var(--font-sans) !important; padding: 10px 14px !important;
}

[data-testid="stSelectbox"] > div > div {
  background: var(--bg3) !important; border: 1px solid var(--border) !important;
  border-radius: var(--r-md) !important; color: var(--text0) !important; font-size: 13px !important;
}

[data-testid="stMetric"] {
  background: var(--bg3) !important; border-radius: var(--r-md) !important;
  padding: 12px !important; border: 1px solid var(--border) !important;
}
[data-testid="stMetric"] label { color: var(--text2) !important; font-size: 11px !important; font-family: var(--font-mono) !important; }
[data-testid="stMetricValue"] { color: var(--accent) !important; font-size: 22px !important; font-weight: 600 !important; }

[data-testid="stDownloadButton"] > button {
  background: var(--bg3) !important; border: 1px solid var(--border-s) !important;
  color: var(--text1) !important; border-radius: var(--r-md) !important;
  font-size: 13px !important; width: 100%;
}
[data-testid="stDownloadButton"] > button:hover { border-color: var(--green) !important; color: var(--green) !important; }

[data-testid="stFileUploader"] {
  background: var(--bg2) !important; border: 1.5px dashed var(--border-s) !important;
  border-radius: var(--r-lg) !important;
}

.row-u { display: flex; justify-content: flex-end; margin: 12px 0 4px; gap: 10px; align-items: flex-end; }
.row-a { display: flex; justify-content: flex-start; margin: 4px 0 12px; gap: 10px; align-items: flex-start; }
.av { width: 30px; height: 30px; border-radius: 50%; flex-shrink: 0; display: flex; align-items: center; justify-content: center; font-size: 13px; border: 1px solid var(--border); }
.av-a { background: rgba(124,131,253,0.12); }
.av-u { background: rgba(52,211,153,0.10); }
.bub-u { background: var(--accent); color: #fff; padding: 10px 15px; border-radius: 18px 18px 5px 18px; font-size: 13.5px; line-height: 1.65; max-width: 72%; font-family: var(--font-sans); }
.bub-a { background: var(--bg3); border: 1px solid var(--border); color: var(--text0); padding: 12px 16px; border-radius: 5px 18px 18px 18px; font-size: 13.5px; line-height: 1.8; max-width: 84%; font-family: var(--font-sans); }
.bub-a strong { color: var(--text0); }
.bub-a code { background: var(--bg0); color: #93C5FD; padding: 1px 6px; border-radius: 4px; font-family: var(--font-mono); font-size: 12px; }

.conf { display: inline-flex; align-items: center; gap: 4px; font-size: 10px; font-family: var(--font-mono); padding: 2px 8px; border-radius: 20px; margin-left: 8px; vertical-align: middle; font-weight: 500; }
.c-hi { background: rgba(52,211,153,.12); color: #34D399; border: 1px solid rgba(52,211,153,.25); }
.c-md { background: rgba(251,191,36,.10); color: #FBBF24; border: 1px solid rgba(251,191,36,.22); }
.c-lo { background: rgba(248,113,113,.10); color: #F87171; border: 1px solid rgba(248,113,113,.22); }

.fol-chip { display: inline-block; background: var(--bg4); border: 1px solid var(--border); color: var(--text1); border-radius: 20px; padding: 4px 12px; font-size: 12px; margin: 3px 4px 3px 0; cursor: pointer; }

.src-wrap { margin-top: 12px; padding-top: 10px; border-top: 1px solid var(--border); }
.src-title { font-size: 10px; font-weight: 600; color: var(--text2); text-transform: uppercase; letter-spacing: .8px; margin-bottom: 7px; font-family: var(--font-mono); }
.src-item { background: var(--bg0); border: 1px solid var(--border); border-radius: var(--r-sm); padding: 8px 11px; margin-bottom: 5px; font-size: 11.5px; color: var(--text1); font-family: var(--font-mono); line-height: 1.55; }
.src-n { background: var(--accent-d); color: #fff; font-size: 9px; padding: 1px 6px; border-radius: 4px; margin-right: 6px; }
.src-pg { float: right; font-size: 9px; color: var(--text2); background: var(--bg3); padding: 1px 6px; border-radius: 3px; }

.hero-card { background: linear-gradient(135deg, rgba(124,131,253,.12) 0%, rgba(93,99,220,.05) 100%); border: 1px solid rgba(124,131,253,.3); border-radius: var(--r-lg); padding: 20px 24px; margin-bottom: 18px; }
.hero-eyebrow { font-size: 9px; font-weight: 600; color: var(--accent); text-transform: uppercase; letter-spacing: 1.2px; font-family: var(--font-mono); margin-bottom: 8px; }
.hero-text { font-size: 16px; color: var(--text0); font-weight: 500; line-height: 1.65; margin: 0; }

.info-card { background: var(--bg3); border: 1px solid var(--border); border-left: 3px solid var(--accent); border-radius: 0 var(--r-md) var(--r-md) 0; padding: 14px 18px; margin-bottom: 10px; }
.info-eyebrow { font-size: 9px; font-weight: 600; color: var(--accent); text-transform: uppercase; letter-spacing: .9px; font-family: var(--font-mono); margin-bottom: 5px; }
.info-text { font-size: 13.5px; color: var(--text1); line-height: 1.75; }

.tag { display: inline-block; background: rgba(124,131,253,.1); border: 1px solid rgba(124,131,253,.25); color: #A5B4FC; border-radius: var(--r-sm); padding: 3px 10px; font-size: 12px; margin: 3px 3px 3px 0; font-family: var(--font-mono); }
.tag-green { background: rgba(52,211,153,.08); border: 1px solid rgba(52,211,153,.2); color: #6EE7B7; }
.tag-amber { background: rgba(251,191,36,.08); border: 1px solid rgba(251,191,36,.2); color: #FCD34D; }

.sec-head { font-size: 11px; font-weight: 600; color: var(--text2); text-transform: uppercase; letter-spacing: .9px; font-family: var(--font-mono); margin: 18px 0 10px; border-bottom: 1px solid var(--border); padding-bottom: 6px; }

.cmp-card { background: var(--bg3); border: 1px solid var(--border); border-radius: var(--r-md); padding: 16px; }
.cmp-head { font-size: 12px; font-weight: 600; color: var(--text0); border-bottom: 1px solid var(--border); padding-bottom: 8px; margin-bottom: 12px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.cmp-lbl { font-size: 9px; color: var(--accent); font-family: var(--font-mono); text-transform: uppercase; letter-spacing: .6px; margin-bottom: 4px; }
.cmp-val { font-size: 13px; color: var(--text1); line-height: 1.7; margin-bottom: 12px; }

.sb-label { font-size: 9.5px; font-weight: 600; color: var(--text3); text-transform: uppercase; letter-spacing: .9px; font-family: var(--font-mono); margin: 14px 0 7px; display: block; }
.logo-wrap { padding: 18px 0 14px; border-bottom: 1px solid var(--border); margin-bottom: 4px; }
.logo-name { font-size: 18px; font-weight: 700; color: var(--text0); letter-spacing: -.5px; margin: 0; }
.logo-name em { color: var(--accent); font-style: normal; }
.logo-tag { font-size: 10px; color: var(--text2); font-family: var(--font-mono); margin-top: 3px; }

.flashcard { background: var(--bg3); border: 1px solid var(--border-s); border-radius: var(--r-lg); padding: 20px 22px; margin-bottom: 10px; }

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg0); }
::-webkit-scrollbar-thumb { background: var(--bg4); border-radius: 3px; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"], [data-testid="stDecoration"], .stDeployButton { display: none; }
hr { border-color: var(--border) !important; margin: 10px 0 !important; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════
for k, v in {
    "messages": [], "collections": {}, "active_col": None,
    "summaries": {}, "compare_cache": {},
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ═══════════════════════════════════════════════════════════════════════════════
# RAG ENGINE  — LCEL pipeline (no legacy Chain classes)
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading embedding model…")
def _load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

def _llm(api_key: str, temp: float = 0.15):
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=temp,
    )

def _ingest(uploaded_file) -> dict:
    ext = Path(uploaded_file.name).suffix.lower()
    if ext not in {".pdf", ".txt", ".md"}:
        raise ValueError(f"Unsupported file type '{ext}'. Use PDF, TXT or MD.")
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(uploaded_file.read())
        path = tmp.name
    try:
        docs = PyPDFLoader(path).load() if ext == ".pdf" else TextLoader(path, encoding="utf-8").load()
    finally:
        os.unlink(path)
    if not docs:
        raise ValueError("No text extracted — ensure the PDF contains selectable text.")
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=900, chunk_overlap=180,
        separators=["\n\n", "\n", ". ", "! ", "? ", " "],
    ).split_documents(docs)
    if not chunks:
        raise ValueError("Document produced no usable chunks.")
    col_id = hashlib.md5((uploaded_file.name + str(len(chunks))).encode()).hexdigest()[:8]
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=_load_embeddings(),
        collection_name=col_id,
    )
    return {"filename": uploaded_file.name, "chunks": len(chunks),
            "vectordb": vectordb, "col_id": col_id, "pages": len(docs)}

# ── LCEL RAG chain (replaces RetrievalQA) ────────────────────────────────────
_RAG_TEMPLATE = """You are ScholarMind, a precise AI research assistant.
Answer using ONLY the retrieved context below. Be structured and cite evidence.
If context is insufficient, say so — never fabricate.

Retrieved context:
{context}

Question: {question}

Answer:"""

_RAG_PROMPT = ChatPromptTemplate.from_template(_RAG_TEMPLATE)

def _format_docs(docs: list) -> str:
    return "\n\n---\n\n".join(d.page_content for d in docs)

def _rag_query(question: str, vectordb, api_key: str, top_k: int) -> dict:
    retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": top_k, "fetch_k": max(top_k * 3, 12)},
    )
    # LCEL chain — no legacy Chain class
    chain = (
        {"context": retriever | RunnableLambda(_format_docs), "question": RunnablePassthrough()}
        | _RAG_PROMPT
        | _llm(api_key)
        | StrOutputParser()
    )
    answer = chain.invoke(question)
    # Retrieve sources separately for display
    source_docs = retriever.invoke(question)
    seen, sources = set(), []
    for doc in source_docs:
        txt = doc.page_content.strip()
        if txt in seen:
            continue
        seen.add(txt)
        sources.append({
            "content": txt[:450] + ("…" if len(txt) > 450 else ""),
            "page": doc.metadata.get("page", "–"),
        })
    hedges = ["don't have", "not enough", "cannot", "no information", "not provided"]
    conf = "low" if any(h in answer.lower() for h in hedges) \
        else "medium" if len(answer.strip()) < 130 else "high"
    return {"answer": answer, "sources": sources, "confidence": conf}

# ── LLM-only calls (summary, compare, etc.) ───────────────────────────────────
def _llm_json(prompt: str, api_key: str, temp: float = 0.1) -> dict | list:
    raw = _llm(api_key, temp).invoke(prompt)
    text = raw.content if hasattr(raw, "content") else str(raw)
    text = re.sub(r"```json|```", "", text).strip()
    return json.loads(text)

_FOLLOWUP_TMPL = """Given this Q&A about a research paper, suggest 3 follow-up questions a researcher would ask.
Return ONLY a JSON array of 3 strings. No explanation.

Q: {q}
A: {a}

JSON:"""

_SUMMARY_TMPL = """Analyze this research paper excerpt and return a JSON object with exactly these keys:
"one_liner", "problem", "method", "results", "contribution", "limitations",
"keywords" (array of 8-10 strings), "paper_type" (one of empirical/theoretical/survey/system/position),
"venue_guess", "year_guess".
Return ONLY valid JSON, no markdown fences.

Paper excerpt:
{ctx}"""

_COMPARE_TMPL = """Compare two research papers. Return JSON with keys:
"a_contribution", "b_contribution", "similarities" (array of 3),
"differences" (array of 4), "complementary", "read_order", "a_strength", "b_strength".
Return ONLY valid JSON.

Paper A ({na}):
{ca}

Paper B ({nb}):
{cb}"""

_FLASHCARD_TMPL = """Create 6 study flashcards from this paper.
Return a JSON array of objects each with "q" and "a" keys.
Return ONLY a valid JSON array, no markdown fences.

Paper:
{ctx}"""

_GAP_TMPL = """Identify from this research paper:
Return JSON with keys: "explicit_gaps" (array of 3-4), "implicit_gaps" (array of 3-4),
"future_directions" (array of 3-4), "open_questions" (array of 3-4).
Return ONLY valid JSON.

Paper:
{ctx}"""

_ELI15_TMPL = "Explain this research paper to a curious 15-year-old using simple analogies. Max 180 words.\n\nPaper:\n{ctx}"

def _sample_docs(vectordb, n_head=5, n_mid=4, n_tail=3) -> str:
    all_texts = vectordb.get().get("documents", [])
    mid = len(all_texts) // 2
    sample = all_texts[:n_head] + all_texts[mid:mid+n_mid] + all_texts[-n_tail:]
    return "\n\n---\n\n".join(sample)[:6000]

def _get_summary(col_id, vectordb, api_key):
    if col_id not in st.session_state.summaries:
        ctx = _sample_docs(vectordb)
        data = _llm_json(_SUMMARY_TMPL.format(ctx=ctx), api_key)
        if not isinstance(data, dict):
            data = {}
        st.session_state.summaries[col_id] = data
    return st.session_state.summaries[col_id]

def _get_flashcards(col_id, vectordb, api_key):
    key = f"fc_{col_id}"
    if key not in st.session_state:
        ctx = _sample_docs(vectordb)
        data = _llm_json(_FLASHCARD_TMPL.format(ctx=ctx), api_key, temp=0.3)
        st.session_state[key] = data if isinstance(data, list) else []
    return st.session_state[key]

def _get_gaps(col_id, vectordb, api_key):
    key = f"gaps_{col_id}"
    if key not in st.session_state:
        texts = vectordb.get().get("documents", [])
        mid = len(texts) // 2
        ctx = "\n\n".join(texts[mid:] + texts[-4:])[:5500]
        data = _llm_json(_GAP_TMPL.format(ctx=ctx), api_key)
        st.session_state[key] = data if isinstance(data, dict) else {}
    return st.session_state[key]

def _get_eli15(col_id, vectordb, api_key):
    key = f"eli15_{col_id}"
    if key not in st.session_state:
        ctx = "\n\n".join(vectordb.get().get("documents", [])[:6])[:4000]
        raw = _llm(api_key, 0.5).invoke(_ELI15_TMPL.format(ctx=ctx))
        st.session_state[key] = raw.content if hasattr(raw, "content") else str(raw)
    return st.session_state[key]

def _get_compare(id_a, id_b, api_key):
    key = f"{id_a}__{id_b}"
    rkey = f"{id_b}__{id_a}"
    if key not in st.session_state.compare_cache and rkey not in st.session_state.compare_cache:
        ma, mb = st.session_state.collections[id_a], st.session_state.collections[id_b]
        da = "\n\n".join(ma["vectordb"].get().get("documents", [])[:8])[:3000]
        db = "\n\n".join(mb["vectordb"].get().get("documents", [])[:8])[:3000]
        data = _llm_json(_COMPARE_TMPL.format(
            na=ma["filename"], ca=da, nb=mb["filename"], cb=db), api_key)
        st.session_state.compare_cache[key] = data if isinstance(data, dict) else {}
    return st.session_state.compare_cache.get(key) or st.session_state.compare_cache.get(rkey, {})

def _get_followups(q, a, api_key):
    try:
        data = _llm_json(_FOLLOWUP_TMPL.format(q=q, a=a[:600]), api_key, temp=0.4)
        return data[:3] if isinstance(data, list) else []
    except Exception:
        return []

# ── Export ─────────────────────────────────────────────────────────────────────
def _export_chat(messages, filename):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"# ScholarMind Session\n**Paper:** {filename}  \n**Date:** {ts}\n\n---"]
    for m in messages:
        if m["role"] == "user":
            lines.append(f"\n### ❓ {m['content']}\n")
        else:
            lines.append(f"\n**Answer** *(conf: {m.get('confidence','')})*\n\n{m['content']}\n")
            for i, s in enumerate(m.get("sources", []), 1):
                lines.append(f"> [{i}] p.{s['page']} — {s['content']}\n")
    return "\n".join(lines)

def _export_summary(s, filename):
    return f"""# Summary — {filename}
*ScholarMind · {datetime.datetime.now().strftime('%Y-%m-%d')}*

## In One Line
{s.get('one_liner','')}

## Problem
{s.get('problem','')}

## Methodology
{s.get('method','')}

## Key Results
{s.get('results','')}

## Contribution
{s.get('contribution','')}

## Limitations
{s.get('limitations','')}

## Keywords
{', '.join(s.get('keywords', []))}

---
*Type: {s.get('paper_type','')} · Venue: {s.get('venue_guess','')} · Year: {s.get('year_guess','')}*
"""

# ── UI helpers ─────────────────────────────────────────────────────────────────
def _conf_badge(conf):
    m = {"high": ("● HIGH", "c-hi"), "medium": ("◑ MED", "c-md"), "low": ("○ LOW", "c-lo")}
    lbl, cls = m.get(conf, ("? UNK", "c-lo"))
    return f'<span class="conf {cls}">{lbl}</span>'

def _render_msg(msg):
    if msg["role"] == "user":
        st.markdown(
            f'<div class="row-u"><div class="bub-u">{msg["content"]}</div>'
            f'<div class="av av-u">👤</div></div>', unsafe_allow_html=True)
    else:
        ans = msg["content"].replace("\n", "<br>")
        srcs = ""
        if msg.get("sources"):
            items = "".join(
                f'<div class="src-item"><span class="src-n">{i+1}</span>'
                f'<span class="src-pg">p.{s["page"]}</span>{s["content"]}</div>'
                for i, s in enumerate(msg["sources"]))
            srcs = (f'<div class="src-wrap"><div class="src-title">'
                    f'📎 Evidence · {len(msg["sources"])} chunks</div>{items}</div>')
        fols = ""
        if msg.get("follow_ups"):
            chips = "".join(f'<span class="fol-chip">{q}</span>' for q in msg["follow_ups"])
            fols = (f'<div style="margin-top:10px;padding-top:8px;border-top:1px solid var(--border)">'
                    f'<div style="font-size:10px;color:var(--text2);font-family:var(--font-mono);margin-bottom:5px;">💡 FOLLOW-UPS</div>'
                    f'{chips}</div>')
        st.markdown(
            f'<div class="row-a"><div class="av av-a">🔬</div>'
            f'<div class="bub-a">{ans}{_conf_badge(msg.get("confidence","medium"))}{srcs}{fols}</div></div>',
            unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="logo-wrap"><p class="logo-name">Scholar<em>Mind</em></p>'
                '<p class="logo-tag">Gemini 2.0 · LCEL RAG · Research AI</p></div>',
                unsafe_allow_html=True)

    st.markdown('<span class="sb-label">🔑 Gemini API Key</span>', unsafe_allow_html=True)
    api_key = st.text_input("key", type="password", placeholder="AIzaSy…",
                             label_visibility="collapsed",
                             help="Free at aistudio.google.com/app/apikey")
    if not api_key:
        st.markdown('<p style="font-size:11px;color:#FBBF24;margin:2px 0 0;">⚡ Required to use the app</p>',
                    unsafe_allow_html=True)

    st.divider()
    st.markdown('<span class="sb-label">📄 Upload Paper</span>', unsafe_allow_html=True)
    uploaded = st.file_uploader("up", type=["pdf", "txt", "md"],
                                 label_visibility="collapsed")
    if uploaded and api_key:
        existing = {v["filename"] for v in st.session_state.collections.values()}
        if uploaded.name not in existing:
            with st.spinner("Indexing…"):
                try:
                    res = _ingest(uploaded)
                    st.session_state.collections[res["col_id"]] = res
                    st.session_state.active_col = res["col_id"]
                    st.session_state.messages = []
                    st.success(f"✓ {res['chunks']} chunks indexed")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed: {e}")
        else:
            cid = next(k for k, v in st.session_state.collections.items() if v["filename"] == uploaded.name)
            if st.session_state.active_col != cid:
                st.session_state.active_col = cid
                st.session_state.messages = []
                st.rerun()
    elif uploaded and not api_key:
        st.warning("Add API key first.")

    st.divider()
    st.markdown('<span class="sb-label">📚 Paper Library</span>', unsafe_allow_html=True)
    if not st.session_state.collections:
        st.markdown('<p style="font-size:12px;color:var(--text3);text-align:center;padding:8px 0;">No papers yet</p>',
                    unsafe_allow_html=True)
    else:
        for cid, meta in list(st.session_state.collections.items()):
            active = cid == st.session_state.active_col
            c1, c2 = st.columns([8, 1])
            short = (meta["filename"][:26] + "…") if len(meta["filename"]) > 26 else meta["filename"]
            if c1.button(f"{'▶ ' if active else ''}{short}", key=f"sel_{cid}",
                          use_container_width=True, type="primary" if active else "secondary"):
                st.session_state.active_col = cid
                st.session_state.messages = []
                st.rerun()
            if c2.button("✕", key=f"del_{cid}"):
                del st.session_state.collections[cid]
                if st.session_state.active_col == cid:
                    st.session_state.active_col = None
                    st.session_state.messages = []
                st.rerun()
            st.markdown(f'<p style="font-size:10px;color:var(--text3);font-family:var(--font-mono);margin:-4px 0 6px 2px;">'
                        f'{meta["chunks"]} chunks · {meta.get("pages","?")}p · {cid}</p>', unsafe_allow_html=True)

    st.divider()
    st.markdown('<span class="sb-label">⚙️ Retrieval</span>', unsafe_allow_html=True)
    top_k = st.slider("Chunks (k)", 2, 8, 4)
    show_src = st.checkbox("Show evidence chunks", value=True)

    st.divider()
    st.markdown('<p style="font-size:10px;color:var(--text3);text-align:center;font-family:var(--font-mono);">'
                'Gemini 2.0 Flash · MiniLM-L6-v2<br>LangChain LCEL · ChromaDB 0.6</p>',
                unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
active_meta = st.session_state.collections.get(st.session_state.active_col)

if not active_meta:
    st.markdown('<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding:80px 40px;text-align:center;">'
                '<div style="font-size:52px;margin-bottom:18px;">🔬</div>'
                '<p style="font-size:20px;font-weight:600;color:#EEF0F8;margin-bottom:10px;letter-spacing:-.3px;">ScholarMind</p>'
                '<p style="font-size:13.5px;color:#A0ABCC;line-height:1.75;max-width:400px;">'
                'Upload a research paper to unlock 7 analysis modes:<br>'
                '<strong>Chat · Summary · Concepts · Gaps · Flashcards · Compare · Export</strong></p>'
                '</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    for i, (icon, title, desc) in enumerate([
        ("💬", "Grounded Chat", "LCEL RAG with confidence scores"),
        ("📋", "Auto-Summary", "Problem · Method · Results · Contribution"),
        ("🔍", "Research Gaps", "Explicit + implicit gaps & future work"),
        ("📚", "Flashcards", "Auto-generated study cards + ELI15"),
        ("🏷️", "Concepts", "Keywords + frequency chart"),
        ("⚖️", "Compare", "Side-by-side paper analysis"),
        ("💡", "Follow-ups", "Auto-suggested next questions"),
        ("📤", "Export", "MD · JSON downloads"),
    ]):
        with cols[i % 4]:
            st.markdown(f'<div style="background:var(--bg2);border:1px solid var(--border);border-radius:var(--r-md);padding:14px;margin-bottom:10px;text-align:center;">'
                        f'<div style="font-size:22px;margin-bottom:6px;">{icon}</div>'
                        f'<div style="font-size:12.5px;font-weight:600;color:var(--text0);margin-bottom:4px;">{title}</div>'
                        f'<div style="font-size:11px;color:var(--text2);line-height:1.5;">{desc}</div></div>',
                        unsafe_allow_html=True)
    st.stop()

# Top bar metrics
h1, h2, h3, h4 = st.columns([5, 1, 1, 1])
with h1:
    st.markdown(f'<p style="font-size:11px;color:var(--text3);font-family:var(--font-mono);margin-bottom:1px;">ACTIVE PAPER</p>'
                f'<p style="font-size:17px;font-weight:700;color:var(--text0);margin:0;letter-spacing:-.3px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">'
                f'{active_meta["filename"]}</p>', unsafe_allow_html=True)
h2.metric("Chunks", active_meta["chunks"])
h3.metric("Pages", active_meta.get("pages", "?"))
h4.metric("k", top_k)
st.markdown('<hr style="margin:10px 0 0 !important;">', unsafe_allow_html=True)

# ── TABS ───────────────────────────────────────────────────────────────────────
t_chat, t_sum, t_conc, t_gaps, t_flash, t_cmp, t_exp = st.tabs([
    "💬 Chat", "📋 Summary", "🏷️ Concepts",
    "🔍 Research Gaps", "📚 Flashcards", "⚖️ Compare", "📤 Export",
])

# ── TAB 1: CHAT ────────────────────────────────────────────────────────────────
with t_chat:
    if not st.session_state.messages:
        c1, c2, c3 = st.columns(3)
        for i, q in enumerate([
            "What is the main contribution?", "Summarize the methodology",
            "What are the key quantitative results?", "What datasets were used?",
            "What are the main limitations?", "How does this compare to prior work?",
        ]):
            if [c1, c2, c3][i % 3].button(q, key=f"chip_{i}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": q})
                st.rerun()
        st.markdown("<br>", unsafe_allow_html=True)

    for msg in st.session_state.messages:
        _render_msg(msg)

    if not api_key:
        st.info("🔑 Enter your Gemini API key in the sidebar.")
    else:
        question = st.chat_input("Ask anything about this paper…")
        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.spinner("Retrieving evidence · Generating answer…"):
                try:
                    res = _rag_query(question, active_meta["vectordb"], api_key, top_k)
                    fol = _get_followups(question, res["answer"], api_key)
                    st.session_state.messages.append({
                        "role": "assistant", "content": res["answer"],
                        "sources": res["sources"] if show_src else [],
                        "confidence": res["confidence"], "follow_ups": fol,
                    })
                except Exception as e:
                    st.session_state.messages.append({
                        "role": "assistant", "content": f"⚠️ Error: {e}",
                        "sources": [], "confidence": "low", "follow_ups": [],
                    })
            st.rerun()

    if st.session_state.messages:
        st.button("🗑 Clear chat", key="clr",
                  on_click=lambda: st.session_state.update({"messages": []}))

# ── TAB 2: SUMMARY ─────────────────────────────────────────────────────────────
with t_sum:
    col_id = st.session_state.active_col
    if not api_key:
        st.info("🔑 Add API key.")
    elif col_id not in st.session_state.summaries:
        if st.button("📋 Analyse Paper", type="primary", key="gen_sum"):
            with st.spinner("Analysing with Gemini 2.0 Flash…"):
                try:
                    _get_summary(col_id, active_meta["vectordb"], api_key)
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed: {e}")
    else:
        s = st.session_state.summaries[col_id]
        st.markdown(f'<div class="hero-card"><div class="hero-eyebrow">paper in one line</div>'
                    f'<p class="hero-text">{s.get("one_liner","")}</p></div>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Type", s.get("paper_type","?").upper())
        m2.metric("Venue", s.get("venue_guess","?")[:12])
        m3.metric("Year", s.get("year_guess","?"))
        m4.metric("Keywords", len(s.get("keywords",[])))
        st.markdown("<br>", unsafe_allow_html=True)
        for key, label in [("problem","🎯 Problem"),("method","⚙️ Methodology"),
                            ("results","📊 Results"),("contribution","💡 Contribution"),
                            ("limitations","⚠️ Limitations")]:
            if s.get(key):
                st.markdown(f'<div class="info-card"><div class="info-eyebrow">{label}</div>'
                            f'<div class="info-text">{s[key]}</div></div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        if c1.button("🔄 Regenerate", key="regen_sum"):
            st.session_state.summaries.pop(col_id, None)
            st.rerun()
        c2.download_button("⬇️ Download Summary (.md)",
                            data=_export_summary(s, active_meta["filename"]),
                            file_name=f"summary_{col_id}.md", mime="text/markdown", key="dl_sum")

# ── TAB 3: CONCEPTS ────────────────────────────────────────────────────────────
with t_conc:
    col_id = st.session_state.active_col
    if not api_key:
        st.info("🔑 Add API key.")
    elif col_id not in st.session_state.summaries:
        if st.button("🏷️ Extract Concepts", type="primary", key="gen_conc"):
            with st.spinner("Extracting…"):
                try:
                    _get_summary(col_id, active_meta["vectordb"], api_key)
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed: {e}")
    else:
        kws = st.session_state.summaries[col_id].get("keywords", [])
        if kws:
            st.markdown("".join(f'<span class="tag">{k}</span>' for k in kws) + "<br><br>",
                        unsafe_allow_html=True)
            # Frequency chart
            all_texts = " ".join(active_meta["vectordb"].get().get("documents", [])[:20]).lower()
            scores = [max(all_texts.count(k.lower()), 1) for k in kws]
            fig = go.Figure(go.Bar(x=scores, y=kws, orientation="h",
                                   marker=dict(color=scores, colorscale=[[0,"#2D3A52"],[0.5,"#5B5FD9"],[1,"#7C83FD"]], showscale=False),
                                   hovertemplate="%{y}: %{x}<extra></extra>"))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font=dict(family="JetBrains Mono", color="#A0ABCC", size=11),
                              margin=dict(l=0,r=10,t=10,b=10), height=260,
                              xaxis=dict(gridcolor="rgba(99,120,180,0.1)", zeroline=False),
                              yaxis=dict(gridcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.markdown('<div class="sec-head">Deep-dive</div>', unsafe_allow_html=True)
            cols = st.columns(3)
            for i, kw in enumerate(kws[:9]):
                if cols[i % 3].button(f"⚡ {kw}", key=f"kw_{i}", use_container_width=True):
                    q = f"Explain how '{kw}' is used in this paper."
                    st.session_state.messages.append({"role": "user", "content": q})
                    with st.spinner(f"Looking up '{kw}'…"):
                        try:
                            res = _rag_query(q, active_meta["vectordb"], api_key, top_k)
                            st.session_state.messages.append({
                                "role": "assistant", "content": res["answer"],
                                "sources": res["sources"] if show_src else [],
                                "confidence": res["confidence"], "follow_ups": [],
                            })
                        except Exception as e:
                            st.session_state.messages.append({"role": "assistant", "content": f"⚠️ {e}", "sources": [], "confidence": "low", "follow_ups": []})
                    st.info(f"Answer added to Chat tab ↑")

# ── TAB 4: RESEARCH GAPS ───────────────────────────────────────────────────────
with t_gaps:
    col_id = st.session_state.active_col
    gap_key = f"gaps_{col_id}"
    if not api_key:
        st.info("🔑 Add API key.")
    elif gap_key not in st.session_state:
        if st.button("🔍 Find Research Gaps", type="primary", key="gen_gaps"):
            with st.spinner("Analysing gaps…"):
                try:
                    _get_gaps(col_id, active_meta["vectordb"], api_key)
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed: {e}")
    else:
        gaps = st.session_state[gap_key]
        for key, label, cls in [
            ("explicit_gaps", "📌 Explicit Gaps", "tag"),
            ("implicit_gaps", "🔎 Implicit Gaps", "tag-amber"),
            ("future_directions", "🚀 Future Directions", "tag-green"),
            ("open_questions", "❓ Open Questions", "tag"),
        ]:
            items = gaps.get(key, [])
            if items:
                st.markdown(f'<div class="sec-head">{label}</div>', unsafe_allow_html=True)
                st.markdown("".join(f'<span class="tag {cls}">{it}</span>' for it in items),
                            unsafe_allow_html=True)
                for j, item in enumerate(items[:3]):
                    short = (item[:60] + "…") if len(item) > 60 else item
                    if st.button(f"💬 Discuss: {short}", key=f"gap_{key}_{j}", use_container_width=True):
                        q = f"What does the paper say about this gap: '{item}'?"
                        st.session_state.messages.append({"role": "user", "content": q})
                        with st.spinner("Generating…"):
                            try:
                                res = _rag_query(q, active_meta["vectordb"], api_key, top_k)
                                st.session_state.messages.append({
                                    "role": "assistant", "content": res["answer"],
                                    "sources": res["sources"] if show_src else [],
                                    "confidence": res["confidence"], "follow_ups": [],
                                })
                            except Exception as e:
                                st.session_state.messages.append({"role": "assistant", "content": f"⚠️ {e}", "sources": [], "confidence": "low", "follow_ups": []})
                        st.info("Response added to Chat tab ↑")
        if st.button("🔄 Regenerate", key="regen_gaps"):
            st.session_state.pop(gap_key, None)
            st.rerun()

# ── TAB 5: FLASHCARDS ──────────────────────────────────────────────────────────
with t_flash:
    col_id = st.session_state.active_col
    fc_key = f"fc_{col_id}"
    if not api_key:
        st.info("🔑 Add API key.")
    elif fc_key not in st.session_state:
        if st.button("📚 Generate Flashcards", type="primary", key="gen_fc"):
            with st.spinner("Creating flashcards…"):
                try:
                    _get_flashcards(col_id, active_meta["vectordb"], api_key)
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed: {e}")
    else:
        cards = st.session_state[fc_key]
        eli_key = f"eli15_{col_id}"
        head, btn_col = st.columns([3, 1])
        with head:
            st.markdown(f'<div class="sec-head">{len(cards)} Flashcards</div>', unsafe_allow_html=True)
        with btn_col:
            if st.button("🧒 ELI15", key="eli15_btn", use_container_width=True):
                if eli_key not in st.session_state:
                    with st.spinner("Simplifying…"):
                        try:
                            _get_eli15(col_id, active_meta["vectordb"], api_key)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed: {e}")
        if eli_key in st.session_state:
            st.markdown(f'<div class="hero-card"><div class="hero-eyebrow">🧒 Explain Like I\'m 15</div>'
                        f'<p style="font-size:14px;color:var(--text1);line-height:1.8;margin:0;">'
                        f'{st.session_state[eli_key]}</p></div>', unsafe_allow_html=True)
        if cards:
            lc, rc = st.columns(2)
            for i, card in enumerate(cards):
                with (lc if i % 2 == 0 else rc):
                    with st.expander(f"Card {i+1} — {card.get('q','')[:55]}"):
                        st.markdown(f'<div style="font-size:13px;color:var(--text0);line-height:1.7;">{card.get("a","")}</div>',
                                    unsafe_allow_html=True)
        if st.button("🔄 Regenerate", key="regen_fc"):
            for k in [fc_key, f"eli15_{col_id}"]:
                st.session_state.pop(k, None)
            st.rerun()

# ── TAB 6: COMPARE ─────────────────────────────────────────────────────────────
with t_cmp:
    papers = list(st.session_state.collections.items())
    if len(papers) < 2:
        st.markdown(f'<div style="text-align:center;padding:50px 20px;">'
                    f'<div style="font-size:40px;margin-bottom:12px;">⚖️</div>'
                    f'<p style="font-size:15px;color:var(--text0);font-weight:600;">Upload at least 2 papers</p>'
                    f'<p style="font-size:13px;color:var(--text1);">You have {len(papers)} paper indexed.</p>'
                    f'</div>', unsafe_allow_html=True)
    elif not api_key:
        st.info("🔑 Add API key.")
    else:
        pmap = {cid: meta["filename"] for cid, meta in papers}
        ca, cb = st.columns(2)
        sel_a = ca.selectbox("Paper A", list(pmap.keys()), format_func=lambda x: pmap[x], key="cmp_a")
        opts_b = [k for k in pmap if k != sel_a]
        sel_b = cb.selectbox("Paper B", opts_b, format_func=lambda x: pmap[x], key="cmp_b")
        if st.button("⚖️ Compare", type="primary", use_container_width=True, key="run_cmp"):
            with st.spinner("Comparing…"):
                try:
                    _get_compare(sel_a, sel_b, api_key)
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed: {e}")
        cdata = st.session_state.compare_cache.get(f"{sel_a}__{sel_b}") or \
                st.session_state.compare_cache.get(f"{sel_b}__{sel_a}")
        if cdata:
            l, r = st.columns(2)
            l.markdown(f'<div class="cmp-card"><div class="cmp-head">📄 {pmap[sel_a]}</div>'
                       f'<div class="cmp-lbl">Contribution</div><div class="cmp-val">{cdata.get("a_contribution","")}</div>'
                       f'<div class="cmp-lbl">Strength</div><div class="cmp-val">{cdata.get("a_strength","")}</div></div>',
                       unsafe_allow_html=True)
            r.markdown(f'<div class="cmp-card"><div class="cmp-head">📄 {pmap[sel_b]}</div>'
                       f'<div class="cmp-lbl">Contribution</div><div class="cmp-val">{cdata.get("b_contribution","")}</div>'
                       f'<div class="cmp-lbl">Strength</div><div class="cmp-val">{cdata.get("b_strength","")}</div></div>',
                       unsafe_allow_html=True)
            sl, sr = st.columns(2)
            with sl:
                st.markdown("**🤝 Similarities**")
                for s in cdata.get("similarities", []):
                    st.markdown(f"- {s}")
            with sr:
                st.markdown("**🔀 Differences**")
                for d in cdata.get("differences", []):
                    st.markdown(f"- {d}")
            st.markdown(f'<div class="info-card" style="margin-top:16px;">'
                        f'<div class="info-eyebrow">💡 Synthesis & Reading Order</div>'
                        f'<div class="info-text">{cdata.get("complementary","")}</div>'
                        f'<div style="margin-top:8px;font-size:12px;color:var(--accent);font-family:var(--font-mono);">📖 {cdata.get("read_order","")}</div>'
                        f'</div>', unsafe_allow_html=True)

# ── TAB 7: EXPORT ──────────────────────────────────────────────────────────────
with t_exp:
    col_id = st.session_state.active_col
    fname = active_meta["filename"]
    msgs = st.session_state.messages
    e1, e2, e3 = st.columns(3)
    with e1:
        st.markdown('<div class="info-card"><div class="info-eyebrow">💬 Chat Transcript</div>'
                    f'<div class="info-text">{len([m for m in msgs if m["role"]=="user"])} questions with evidence chains</div></div>',
                    unsafe_allow_html=True)
        if msgs:
            st.download_button("⬇️ Chat (.md)", data=_export_chat(msgs, fname),
                               file_name=f"chat_{col_id}.md", mime="text/markdown",
                               key="dl_chat", use_container_width=True)
        else:
            st.caption("No messages yet.")
    with e2:
        s = st.session_state.summaries.get(col_id)
        st.markdown('<div class="info-card"><div class="info-eyebrow">📋 Paper Summary</div>'
                    '<div class="info-text">Structured analysis with keywords and metadata</div></div>',
                    unsafe_allow_html=True)
        if s:
            st.download_button("⬇️ Summary (.md)", data=_export_summary(s, fname),
                               file_name=f"summary_{col_id}.md", mime="text/markdown",
                               key="dl_sum_e", use_container_width=True)
        else:
            st.caption("Generate summary first.")
    with e3:
        st.markdown('<div class="info-card"><div class="info-eyebrow">🗂 Full Session (JSON)</div>'
                    '<div class="info-text">Messages · Summary · Gaps · Flashcards</div></div>',
                    unsafe_allow_html=True)
        session = {
            "exported_at": datetime.datetime.now().isoformat(),
            "paper": fname, "chunks": active_meta["chunks"],
            "messages": [{"role": m["role"], "content": m["content"],
                          "confidence": m.get("confidence"), "sources": m.get("sources", [])} for m in msgs],
            "summary": st.session_state.summaries.get(col_id, {}),
            "gaps": st.session_state.get(f"gaps_{col_id}", {}),
            "flashcards": st.session_state.get(f"fc_{col_id}", []),
        }
        st.download_button("⬇️ Session (.json)", data=json.dumps(session, indent=2),
                           file_name=f"session_{col_id}.json", mime="application/json",
                           key="dl_json", use_container_width=True)
    st.markdown('<div class="sec-head">Session Stats</div>', unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Questions", len([m for m in msgs if m["role"]=="user"]))
    s2.metric("Summary", "✓" if col_id in st.session_state.summaries else "–")
    s3.metric("Gaps", "✓" if f"gaps_{col_id}" in st.session_state else "–")
    s4.metric("Flashcards", len(st.session_state.get(f"fc_{col_id}", [])))
