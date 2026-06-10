"""
ScholarMind — AI Research Paper Assistant
Gemini 2.0 Flash · LangChain 0.3.x · ChromaDB 0.6.x · Streamlit 1.41
Free deployment: Streamlit Community Cloud

Dependency notes (protobuf-conflict-free stack):
  langchain 0.3.27 + langchain-community 0.3.31
  langchain-google-genai 2.1.4  ← uses google-ai-generativelanguage (protobuf<7 OK)
  chromadb 0.6.3                ← no direct protobuf dependency
  All three allow protobuf 4.x–6.x → zero conflict
"""

import os, re, json, hashlib, tempfile, datetime
import streamlit as st
from pathlib import Path

# ── LangChain 0.3.x imports ───────────────────────────────────────────────────
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

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
# DESIGN SYSTEM — CSS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Tokens ─────────────────────────────────────────────────────────────────── */
:root {
  --bg0:       #05080F;
  --bg1:       #0C1220;
  --bg2:       #111C2E;
  --bg3:       #172035;
  --bg4:       #1D2840;
  --border:    rgba(99,120,180,0.13);
  --border-s:  rgba(99,120,180,0.25);
  --accent:    #7C83FD;
  --accent-d:  #5B5FD9;
  --accent-glow: rgba(124,131,253,0.18);
  --green:     #34D399;
  --amber:     #FBBF24;
  --red:       #F87171;
  --text0:     #EEF0F8;
  --text1:     #A0ABCC;
  --text2:     #5A6A8A;
  --text3:     #2D3A52;
  --font-sans: 'Inter', system-ui, sans-serif;
  --font-mono: 'JetBrains Mono', 'Fira Code', monospace;
  --r-sm:      6px;
  --r-md:      10px;
  --r-lg:      14px;
}

/* ── Global ──────────────────────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main .block-container { background: var(--bg0) !important; }

[data-testid="block-container"] {
  padding: 1.25rem 2rem 3rem !important;
  max-width: 1100px;
}

/* ── Sidebar ─────────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: var(--bg1) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebarContent"] { padding: 0 12px 24px !important; }
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] small { color: var(--text1) !important; font-size: 12px !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: var(--text0) !important; }

/* ── Inputs ───────────────────────────────────────────────────────────────────── */
[data-testid="stTextInput"] input {
  background: var(--bg3) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r-md) !important;
  color: var(--text0) !important;
  font-family: var(--font-mono) !important;
  font-size: 12px !important;
}
[data-testid="stTextInput"] input:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px var(--accent-glow) !important;
  outline: none !important;
}

/* ── Buttons ─────────────────────────────────────────────────────────────────── */
.stButton > button {
  background: var(--bg3) !important;
  border: 1px solid var(--border) !important;
  color: var(--text1) !important;
  border-radius: var(--r-md) !important;
  font-family: var(--font-sans) !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  transition: all 0.15s ease !important;
  padding: 6px 14px !important;
}
.stButton > button:hover {
  border-color: var(--accent) !important;
  color: var(--text0) !important;
  background: var(--accent-glow) !important;
}
.stButton > button[kind="primary"] {
  background: var(--accent) !important;
  border-color: var(--accent) !important;
  color: #fff !important;
}
.stButton > button[kind="primary"]:hover {
  background: var(--accent-d) !important;
  border-color: var(--accent-d) !important;
}

/* ── Tabs ─────────────────────────────────────────────────────────────────────── */
[data-testid="stTabs"] [role="tablist"] {
  background: var(--bg1) !important;
  border-bottom: 1px solid var(--border) !important;
  gap: 2px !important;
  padding: 6px 8px 0 !important;
  border-radius: var(--r-lg) var(--r-lg) 0 0;
}
[data-testid="stTabs"] button[role="tab"] {
  background: transparent !important;
  border: none !important;
  color: var(--text2) !important;
  font-family: var(--font-sans) !important;
  font-size: 12.5px !important;
  font-weight: 500 !important;
  padding: 7px 15px !important;
  border-radius: 8px 8px 0 0 !important;
  transition: all 0.15s !important;
  letter-spacing: 0.01em;
}
[data-testid="stTabs"] button[role="tab"]:hover { color: var(--text0) !important; background: var(--bg3) !important; }
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
  color: var(--accent) !important;
  background: var(--bg2) !important;
  border-bottom: 2px solid var(--accent) !important;
}
[data-testid="stTabsContent"] {
  background: var(--bg2) !important;
  border: 1px solid var(--border) !important;
  border-top: none !important;
  border-radius: 0 0 var(--r-lg) var(--r-lg) !important;
  padding: 24px !important;
  min-height: 520px !important;
}

/* ── Expander ─────────────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
  background: var(--bg3) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r-md) !important;
  margin-bottom: 8px !important;
}
[data-testid="stExpander"] summary {
  color: var(--text1) !important;
  font-size: 13px !important;
  font-family: var(--font-sans) !important;
  padding: 10px 14px !important;
}
[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
  padding: 0 14px 12px !important;
}

/* ── Select / radio / checkbox ───────────────────────────────────────────────── */
[data-testid="stSelectbox"] select,
[data-testid="stSelectbox"] > div > div {
  background: var(--bg3) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r-md) !important;
  color: var(--text0) !important;
  font-size: 13px !important;
}
[data-testid="stRadio"] label,
[data-testid="stCheckbox"] label { color: var(--text1) !important; font-size: 13px !important; }

/* ── Slider ──────────────────────────────────────────────────────────────────── */
[data-testid="stSlider"] [data-testid="stThumbValue"] { color: var(--accent) !important; }

/* ── File uploader ───────────────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
  background: var(--bg2) !important;
  border: 1.5px dashed var(--border-s) !important;
  border-radius: var(--r-lg) !important;
}
[data-testid="stFileUploaderDropzone"] { color: var(--text1) !important; }

/* ── Metric ───────────────────────────────────────────────────────────────────── */
[data-testid="stMetric"] { background: var(--bg3) !important; border-radius: var(--r-md) !important; padding: 12px !important; border: 1px solid var(--border) !important; }
[data-testid="stMetric"] label { color: var(--text2) !important; font-size: 11px !important; font-family: var(--font-mono) !important; }
[data-testid="stMetricValue"] { color: var(--accent) !important; font-size: 22px !important; font-weight: 600 !important; }

/* ── Download button ─────────────────────────────────────────────────────────── */
[data-testid="stDownloadButton"] > button {
  background: var(--bg3) !important;
  border: 1px solid var(--border-s) !important;
  color: var(--text1) !important;
  border-radius: var(--r-md) !important;
  font-size: 13px !important;
  width: 100%;
}
[data-testid="stDownloadButton"] > button:hover {
  border-color: var(--green) !important;
  color: var(--green) !important;
}

/* ── Alert / info boxes ──────────────────────────────────────────────────────── */
[data-testid="stAlert"] {
  background: var(--bg3) !important;
  border-radius: var(--r-md) !important;
  font-size: 13px !important;
}

/* ── Chat ─────────────────────────────────────────────────────────────────────── */
.chat-scroll { max-height: 540px; overflow-y: auto; padding: 4px 0 12px; }
.chat-scroll::-webkit-scrollbar { width: 4px; }
.chat-scroll::-webkit-scrollbar-thumb { background: var(--bg4); border-radius: 2px; }

.row-u { display: flex; justify-content: flex-end; margin: 12px 0 4px; gap: 10px; align-items: flex-end; }
.row-a { display: flex; justify-content: flex-start; margin: 4px 0 12px; gap: 10px; align-items: flex-start; }

.av {
  width: 30px; height: 30px; border-radius: 50%; flex-shrink: 0;
  display: flex; align-items: center; justify-content: center;
  font-size: 13px; border: 1px solid var(--border);
}
.av-a { background: rgba(124,131,253,0.12); }
.av-u { background: rgba(52,211,153,0.10); }

.bub-u {
  background: var(--accent);
  color: #fff; padding: 10px 15px;
  border-radius: 18px 18px 5px 18px;
  font-size: 13.5px; line-height: 1.65; max-width: 72%;
  font-family: var(--font-sans);
}
.bub-a {
  background: var(--bg3);
  border: 1px solid var(--border);
  color: var(--text0);
  padding: 12px 16px;
  border-radius: 5px 18px 18px 18px;
  font-size: 13.5px; line-height: 1.8; max-width: 84%;
  font-family: var(--font-sans);
}
.bub-a strong { color: var(--text0); }
.bub-a code {
  background: var(--bg0); color: #93C5FD;
  padding: 1px 6px; border-radius: 4px;
  font-family: var(--font-mono); font-size: 12px;
}
.bub-a em { color: var(--accent); font-style: normal; font-weight: 500; }

.conf {
  display: inline-flex; align-items: center; gap: 4px;
  font-size: 10px; font-family: var(--font-mono);
  padding: 2px 8px; border-radius: 20px;
  margin-left: 8px; vertical-align: middle; font-weight: 500;
}
.c-hi  { background: rgba(52,211,153,.12); color: #34D399; border: 1px solid rgba(52,211,153,.25); }
.c-md  { background: rgba(251,191,36,.10); color: #FBBF24; border: 1px solid rgba(251,191,36,.22); }
.c-lo  { background: rgba(248,113,113,.10); color: #F87171; border: 1px solid rgba(248,113,113,.22); }

.fol-chip {
  display: inline-block;
  background: var(--bg4); border: 1px solid var(--border);
  color: var(--text1); border-radius: 20px;
  padding: 4px 12px; font-size: 12px; margin: 3px 4px 3px 0;
  cursor: pointer; transition: all 0.12s;
}
.fol-chip:hover { border-color: var(--accent); color: var(--accent); }

/* ── Source chunks ───────────────────────────────────────────────────────────── */
.src-wrap { margin-top: 12px; padding-top: 10px; border-top: 1px solid var(--border); }
.src-title {
  font-size: 10px; font-weight: 600; color: var(--text2);
  text-transform: uppercase; letter-spacing: .8px; margin-bottom: 7px;
  font-family: var(--font-mono);
}
.src-item {
  background: var(--bg0); border: 1px solid var(--border);
  border-radius: var(--r-sm); padding: 8px 11px;
  margin-bottom: 5px; font-size: 11.5px; color: var(--text1);
  font-family: var(--font-mono); line-height: 1.55;
}
.src-n {
  background: var(--accent-d); color: #fff;
  font-size: 9px; padding: 1px 6px; border-radius: 4px;
  margin-right: 6px; font-family: var(--font-mono);
}
.src-pg {
  float: right; font-size: 9px; color: var(--text2);
  background: var(--bg3); padding: 1px 6px; border-radius: 3px;
}

/* ── Summary cards ───────────────────────────────────────────────────────────── */
.hero-card {
  background: linear-gradient(135deg, rgba(124,131,253,.12) 0%, rgba(93,99,220,.05) 100%);
  border: 1px solid rgba(124,131,253,.3);
  border-radius: var(--r-lg);
  padding: 20px 24px; margin-bottom: 18px;
}
.hero-eyebrow {
  font-size: 9px; font-weight: 600; color: var(--accent);
  text-transform: uppercase; letter-spacing: 1.2px;
  font-family: var(--font-mono); margin-bottom: 8px;
}
.hero-text { font-size: 16px; color: var(--text0); font-weight: 500; line-height: 1.65; margin: 0; }

.info-card {
  background: var(--bg3); border: 1px solid var(--border);
  border-left: 3px solid var(--accent);
  border-radius: 0 var(--r-md) var(--r-md) 0;
  padding: 14px 18px; margin-bottom: 10px;
}
.info-eyebrow {
  font-size: 9px; font-weight: 600; color: var(--accent);
  text-transform: uppercase; letter-spacing: .9px;
  font-family: var(--font-mono); margin-bottom: 5px;
}
.info-text { font-size: 13.5px; color: var(--text1); line-height: 1.75; }

/* ── Tags / chips ────────────────────────────────────────────────────────────── */
.tag {
  display: inline-block;
  background: rgba(124,131,253,.1); border: 1px solid rgba(124,131,253,.25);
  color: #A5B4FC; border-radius: var(--r-sm);
  padding: 3px 10px; font-size: 12px; margin: 3px 3px 3px 0;
  font-family: var(--font-mono);
}
.tag-green {
  background: rgba(52,211,153,.08); border: 1px solid rgba(52,211,153,.2);
  color: #6EE7B7;
}
.tag-amber {
  background: rgba(251,191,36,.08); border: 1px solid rgba(251,191,36,.2);
  color: #FCD34D;
}

/* ── Starter chips (chat) ─────────────────────────────────────────────────────── */
.chip-grid { display: flex; flex-wrap: wrap; gap: 7px; margin-bottom: 20px; }
.chip-btn {
  background: var(--bg3); border: 1px solid var(--border);
  color: var(--text1); border-radius: 20px;
  padding: 5px 14px; font-size: 12px; cursor: pointer;
  transition: all 0.12s; font-family: var(--font-sans);
  line-height: 1.4;
}
.chip-btn:hover { border-color: var(--accent); color: var(--accent); background: var(--accent-glow); }

/* ── Empty state ─────────────────────────────────────────────────────────────── */
.empty { display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 80px 40px; text-align: center; }
.empty-icon { font-size: 52px; margin-bottom: 18px; }
.empty-title { font-size: 20px; font-weight: 600; color: var(--text0); margin-bottom: 10px; letter-spacing: -.3px; }
.empty-sub { font-size: 13.5px; color: var(--text1); line-height: 1.75; max-width: 400px; }

/* ── Section heading ─────────────────────────────────────────────────────────── */
.sec-head {
  font-size: 11px; font-weight: 600; color: var(--text2);
  text-transform: uppercase; letter-spacing: .9px;
  font-family: var(--font-mono); margin: 18px 0 10px;
  border-bottom: 1px solid var(--border); padding-bottom: 6px;
}

/* ── Compare ─────────────────────────────────────────────────────────────────── */
.cmp-card {
  background: var(--bg3); border: 1px solid var(--border);
  border-radius: var(--r-md); padding: 16px;
}
.cmp-head {
  font-size: 12px; font-weight: 600; color: var(--text0);
  border-bottom: 1px solid var(--border); padding-bottom: 8px; margin-bottom: 12px;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.cmp-lbl { font-size: 9px; color: var(--accent); font-family: var(--font-mono); text-transform: uppercase; letter-spacing: .6px; margin-bottom: 4px; }
.cmp-val { font-size: 13px; color: var(--text1); line-height: 1.7; margin-bottom: 12px; }

/* ── Sidebar labels ──────────────────────────────────────────────────────────── */
.sb-label {
  font-size: 9.5px; font-weight: 600; color: var(--text3);
  text-transform: uppercase; letter-spacing: .9px;
  font-family: var(--font-mono); margin: 14px 0 7px;
  display: block;
}

/* ── Logo ────────────────────────────────────────────────────────────────────── */
.logo-wrap { padding: 18px 0 14px; border-bottom: 1px solid var(--border); margin-bottom: 4px; }
.logo-name { font-size: 18px; font-weight: 700; color: var(--text0); letter-spacing: -.5px; margin: 0; }
.logo-name em { color: var(--accent); font-style: normal; }
.logo-tag { font-size: 10px; color: var(--text2); font-family: var(--font-mono); margin-top: 3px; }

/* ── Flashcard ───────────────────────────────────────────────────────────────── */
.flashcard {
  background: var(--bg3); border: 1px solid var(--border-s);
  border-radius: var(--r-lg); padding: 20px 22px; margin-bottom: 10px;
}
.fc-q { font-size: 13.5px; font-weight: 600; color: var(--text0); margin-bottom: 10px; line-height: 1.6; }
.fc-a { font-size: 13px; color: var(--text1); line-height: 1.75; padding-top: 10px; border-top: 1px solid var(--border); }
.fc-num { font-size: 10px; color: var(--accent); font-family: var(--font-mono); margin-bottom: 6px; }

/* ── Plotly override ─────────────────────────────────────────────────────────── */
.js-plotly-plot .plotly { background: transparent !important; }

/* ── Hide chrome ─────────────────────────────────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"], [data-testid="stDecoration"], .stDeployButton { display: none; }
hr { border-color: var(--border) !important; margin: 10px 0 !important; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════
_defaults = {
    "messages": [],
    "collections": {},      # col_id → {filename, chunks, vectordb}
    "active_col": None,
    "summaries": {},        # col_id → parsed dict
    "compare_cache": {},    # "id_a__id_b" → dict
    "follow_ups": [],       # suggested follow-up questions
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ═══════════════════════════════════════════════════════════════════════════════
# RAG ENGINE
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
        convert_system_message_to_human=True,
    )

def _ingest(uploaded_file) -> dict:
    ext = Path(uploaded_file.name).suffix.lower()
    if ext not in {".pdf", ".txt", ".md"}:
        raise ValueError(f"Unsupported file type '{ext}'.")
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(uploaded_file.read())
        path = tmp.name
    try:
        if ext == ".pdf":
            docs = PyPDFLoader(path).load()
        else:
            docs = TextLoader(path, encoding="utf-8").load()
    finally:
        os.unlink(path)
    if not docs:
        raise ValueError("No text extracted — try a text-based PDF.")
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

# Prompts
_RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are ScholarMind, a precise AI research assistant.
Answer ONLY from the provided context. If the context lacks enough information, say so.
Be structured: use **bold** for key terms, cite evidence inline.
Never fabricate statistics, author names, or claims not in the context.

Context:
{context}

Question: {question}

Answer:""",
)

_FOLLOWUP_PROMPT = """Given this Q&A exchange about a research paper, suggest 3 concise follow-up questions a researcher would naturally ask next.
Return ONLY a JSON array of 3 strings, no explanation.

Q: {question}
A: {answer}

JSON array:"""

_SUMMARY_PROMPT = """Analyze this research paper and return a JSON object with exactly these keys:
"one_liner": single sentence capturing the core idea,
"problem": the problem being solved (2-3 sentences),
"method": the approach/methodology (2-3 sentences),
"results": key quantitative findings (2-3 sentences, include numbers if available),
"contribution": novelty and why it matters (2 sentences),
"limitations": acknowledged weaknesses (2 sentences),
"keywords": array of 8-10 technical terms,
"paper_type": one of ["empirical","theoretical","survey","system","position","workshop"],
"venue_guess": likely conference or journal,
"year_guess": estimated publication year as a string.

Return ONLY valid JSON. No markdown fences.

Paper content (excerpt):
{context}"""

_COMPARE_PROMPT = """Compare two research papers. Return JSON with:
"a_contribution": Paper A main contribution (2 sentences),
"b_contribution": Paper B main contribution (2 sentences),
"similarities": array of 3 similarity strings,
"differences": array of 4 key difference strings,
"complementary": how they complement each other (1-2 sentences),
"read_order": which to read first and why (1 sentence),
"a_strength": Paper A's key strength (1 sentence),
"b_strength": Paper B's key strength (1 sentence).

Return ONLY valid JSON. No markdown fences.

Paper A ({name_a}):
{ctx_a}

Paper B ({name_b}):
{ctx_b}"""

_FLASHCARD_PROMPT = """Create 6 study flashcards from this research paper.
Return a JSON array of objects, each with "q" (question) and "a" (answer).
Questions should test understanding of: core concept, methodology, key result,
a definition, a limitation, and one surprising finding.
Return ONLY a valid JSON array. No markdown fences.

Paper content:
{context}"""

_GAP_PROMPT = """Based on this research paper, identify:
1. Research gaps the authors explicitly mention
2. Implicit gaps you notice (limitations not fully addressed)
3. Future research directions suggested
4. Open questions this paper raises

Return JSON with keys: "explicit_gaps" (array), "implicit_gaps" (array),
"future_directions" (array), "open_questions" (array).
Each array has 3-4 string items.
Return ONLY valid JSON. No markdown fences.

Paper content:
{context}"""

_ELI5_PROMPT = """Explain this research paper as if explaining to a curious 15-year-old with no technical background.
Use simple analogies, avoid jargon, and make it engaging. Max 200 words.

Paper content:
{context}"""

def _rag_query(question: str, vectordb, api_key: str, top_k: int) -> dict:
    retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": top_k, "fetch_k": max(top_k * 3, 12)},
    )
    chain = RetrievalQA.from_chain_type(
        llm=_llm(api_key),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": _RAG_PROMPT},
    )
    res = chain.invoke({"query": question})
    answer = res["result"]
    seen, sources = set(), []
    for doc in res["source_documents"]:
        txt = doc.page_content.strip()
        if txt in seen: continue
        seen.add(txt)
        sources.append({
            "content": txt[:450] + ("…" if len(txt) > 450 else ""),
            "page": doc.metadata.get("page", "–"),
        })
    hedges = ["don't have", "not enough", "cannot", "no information", "not provided", "unclear"]
    conf = "low" if any(h in answer.lower() for h in hedges) \
        else "medium" if len(answer.strip()) < 130 else "high"
    return {"answer": answer, "sources": sources, "confidence": conf}

def _get_follow_ups(question: str, answer: str, api_key: str) -> list[str]:
    try:
        llm = _llm(api_key, temp=0.4)
        raw = llm.invoke(_FOLLOWUP_PROMPT.format(question=question, answer=answer[:800]))
        text = raw.content if hasattr(raw, "content") else str(raw)
        text = re.sub(r"```json|```", "", text).strip()
        data = json.loads(text)
        return data[:3] if isinstance(data, list) else []
    except Exception:
        return []

def _parse_json_llm(raw) -> dict | list:
    text = raw.content if hasattr(raw, "content") else str(raw)
    text = re.sub(r"```json|```", "", text).strip()
    return json.loads(text)

def _get_summary(col_id: str, vectordb, api_key: str) -> dict:
    if col_id in st.session_state.summaries:
        return st.session_state.summaries[col_id]
    docs = vectordb.get()
    # Sample spread across doc for better coverage
    all_texts = docs.get("documents", [])
    sample = all_texts[:4] + all_texts[len(all_texts)//2:len(all_texts)//2+4] + all_texts[-3:]
    context = "\n\n---\n\n".join(sample)[:6500]
    raw = _llm(api_key, temp=0.1).invoke(_SUMMARY_PROMPT.format(context=context))
    try:
        data = _parse_json_llm(raw)
    except Exception:
        data = {"one_liner": "Summary unavailable.", "problem": "", "method": "",
                "results": "", "contribution": "", "limitations": "",
                "keywords": [], "paper_type": "empirical",
                "venue_guess": "Unknown", "year_guess": "?"}
    st.session_state.summaries[col_id] = data
    return data

def _get_flashcards(col_id: str, vectordb, api_key: str) -> list:
    key = f"fc_{col_id}"
    if key in st.session_state:
        return st.session_state[key]
    docs = vectordb.get()
    texts = docs.get("documents", [])
    sample = texts[:5] + texts[len(texts)//2:len(texts)//2+3]
    context = "\n\n".join(sample)[:5500]
    raw = _llm(api_key, temp=0.3).invoke(_FLASHCARD_PROMPT.format(context=context))
    try:
        cards = _parse_json_llm(raw)
        cards = cards if isinstance(cards, list) else []
    except Exception:
        cards = []
    st.session_state[key] = cards
    return cards

def _get_gaps(col_id: str, vectordb, api_key: str) -> dict:
    key = f"gaps_{col_id}"
    if key in st.session_state:
        return st.session_state[key]
    docs = vectordb.get()
    texts = docs.get("documents", [])
    sample = texts[len(texts)//2:] + texts[-4:]
    context = "\n\n".join(sample)[:5500]
    raw = _llm(api_key, temp=0.2).invoke(_GAP_PROMPT.format(context=context))
    try:
        data = _parse_json_llm(raw)
    except Exception:
        data = {"explicit_gaps": [], "implicit_gaps": [], "future_directions": [], "open_questions": []}
    st.session_state[key] = data
    return data

def _get_eli5(col_id: str, vectordb, api_key: str) -> str:
    key = f"eli5_{col_id}"
    if key in st.session_state:
        return st.session_state[key]
    docs = vectordb.get()
    texts = docs.get("documents", [])
    context = "\n\n".join(texts[:6])[:4000]
    raw = _llm(api_key, temp=0.5).invoke(_ELI5_PROMPT.format(context=context))
    result = raw.content if hasattr(raw, "content") else str(raw)
    st.session_state[key] = result
    return result

def _get_compare(id_a: str, id_b: str, api_key: str) -> dict:
    key = f"{id_a}__{id_b}"
    rkey = f"{id_b}__{id_a}"
    cached = st.session_state.compare_cache.get(key) or st.session_state.compare_cache.get(rkey)
    if cached: return cached
    ma = st.session_state.collections[id_a]
    mb = st.session_state.collections[id_b]
    da = ma["vectordb"].get().get("documents", [])[:8]
    db = mb["vectordb"].get().get("documents", [])[:8]
    raw = _llm(api_key, temp=0.15).invoke(
        _COMPARE_PROMPT.format(
            name_a=ma["filename"], ctx_a="\n\n".join(da)[:3000],
            name_b=mb["filename"], ctx_b="\n\n".join(db)[:3000],
        )
    )
    try:
        data = _parse_json_llm(raw)
    except Exception:
        data = {"a_contribution": "N/A", "b_contribution": "N/A",
                "similarities": [], "differences": [],
                "complementary": "N/A", "read_order": "N/A",
                "a_strength": "N/A", "b_strength": "N/A"}
    st.session_state.compare_cache[key] = data
    return data

# ─── Export helpers ────────────────────────────────────────────────────────────
def _export_md(messages: list, filename: str) -> str:
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"# ScholarMind — Research Session", f"**Paper:** {filename}  ", f"**Date:** {ts}\n\n---\n"]
    for m in messages:
        if m["role"] == "user":
            lines.append(f"\n### ❓ {m['content']}\n")
        else:
            conf = m.get("confidence", "")
            lines.append(f"\n**Answer** *(confidence: {conf})*\n\n{m['content']}\n")
            if m.get("sources"):
                lines.append("\n**Evidence:**\n")
                for i, s in enumerate(m["sources"], 1):
                    lines.append(f"> [{i}] p.{s['page']} — {s['content']}\n")
    return "\n".join(lines)

def _export_summary_md(s: dict, filename: str) -> str:
    kws = ", ".join(s.get("keywords", []))
    return f"""# Paper Analysis — {filename}
*ScholarMind · {datetime.datetime.now().strftime('%Y-%m-%d')}*

## In One Line
{s.get('one_liner','')}

## Problem
{s.get('problem','')}

## Methodology
{s.get('method','')}

## Key Results
{s.get('results','')}

## Contribution & Novelty
{s.get('contribution','')}

## Limitations
{s.get('limitations','')}

## Keywords
{kws}

---
*Type: {s.get('paper_type','')} · Venue: {s.get('venue_guess','')} · Year: {s.get('year_guess','')}*
"""

# ─── UI helpers ───────────────────────────────────────────────────────────────
def _conf_badge(conf: str) -> str:
    m = {"high": ("● HIGH", "c-hi"), "medium": ("◑ MED", "c-md"), "low": ("○ LOW", "c-lo")}
    lbl, cls = m.get(conf, ("? UNK", "c-lo"))
    return f'<span class="conf {cls}">{lbl}</span>'

def _render_msg(msg: dict):
    if msg["role"] == "user":
        st.markdown(
            f'<div class="row-u">'
            f'<div class="bub-u">{msg["content"]}</div>'
            f'<div class="av av-u">👤</div>'
            f'</div>', unsafe_allow_html=True,
        )
    else:
        answer_html = msg["content"].replace("\n", "<br>")
        conf_html = _conf_badge(msg.get("confidence", "medium"))
        src_html = ""
        if msg.get("sources"):
            items = "".join(
                f'<div class="src-item"><span class="src-n">{i+1}</span>'
                f'<span class="src-pg">p.{s["page"]}</span>{s["content"]}</div>'
                for i, s in enumerate(msg["sources"])
            )
            src_html = (f'<div class="src-wrap">'
                        f'<div class="src-title">📎 Evidence · {len(msg["sources"])} chunks</div>'
                        f'{items}</div>')
        fol_html = ""
        if msg.get("follow_ups"):
            chips = "".join(f'<span class="fol-chip">{q}</span>' for q in msg["follow_ups"])
            fol_html = f'<div style="margin-top:10px;padding-top:8px;border-top:1px solid var(--border)"><div style="font-size:10px;color:var(--text2);font-family:var(--font-mono);margin-bottom:5px;">💡 FOLLOW-UP IDEAS</div>{chips}</div>'
        st.markdown(
            f'<div class="row-a">'
            f'<div class="av av-a">🔬</div>'
            f'<div class="bub-a">{answer_html}{conf_html}{src_html}{fol_html}</div>'
            f'</div>', unsafe_allow_html=True,
        )

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="logo-wrap">
      <p class="logo-name">Scholar<em>Mind</em></p>
      <p class="logo-tag">Gemini 2.0 · RAG · Research AI</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<span class="sb-label">🔑 Gemini API Key</span>', unsafe_allow_html=True)
    api_key = st.text_input("key", type="password", placeholder="AIzaSy…",
                             label_visibility="collapsed",
                             help="Free at aistudio.google.com/app/apikey")
    if not api_key:
        st.markdown('<p style="font-size:11px;color:#FBBF24;margin:2px 0 0;">⚡ API key required</p>',
                    unsafe_allow_html=True)

    st.divider()

    st.markdown('<span class="sb-label">📄 Upload Paper</span>', unsafe_allow_html=True)
    uploaded = st.file_uploader("upload", type=["pdf", "txt", "md"],
                                 label_visibility="collapsed",
                                 help="PDF (text-based), TXT, or Markdown")
    if uploaded and api_key:
        existing = {v["filename"] for v in st.session_state.collections.values()}
        if uploaded.name not in existing:
            with st.spinner(f"Indexing…"):
                try:
                    res = _ingest(uploaded)
                    st.session_state.collections[res["col_id"]] = res
                    st.session_state.active_col = res["col_id"]
                    st.session_state.messages = []
                    st.session_state.follow_ups = []
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
    cols_data = st.session_state.collections
    if not cols_data:
        st.markdown('<p style="font-size:12px;color:var(--text3);text-align:center;padding:8px 0;">No papers yet</p>',
                    unsafe_allow_html=True)
    else:
        for cid, meta in list(cols_data.items()):
            active = cid == st.session_state.active_col
            c1, c2 = st.columns([8, 1])
            name = meta["filename"]
            short = (name[:26] + "…") if len(name) > 26 else name
            lbl = f"▶ {short}" if active else short
            if c1.button(lbl, key=f"sel_{cid}", use_container_width=True,
                          type="primary" if active else "secondary"):
                st.session_state.active_col = cid
                st.session_state.messages = []
                st.session_state.follow_ups = []
                st.rerun()
            if c2.button("✕", key=f"del_{cid}"):
                del st.session_state.collections[cid]
                if st.session_state.active_col == cid:
                    st.session_state.active_col = None
                    st.session_state.messages = []
                st.rerun()
            st.markdown(f'<p style="font-size:10px;color:var(--text3);font-family:var(--font-mono);margin:-4px 0 6px 2px;">{meta["chunks"]} chunks · {meta.get("pages","?")}p · {cid}</p>',
                        unsafe_allow_html=True)

    st.divider()

    st.markdown('<span class="sb-label">⚙️ Retrieval</span>', unsafe_allow_html=True)
    top_k = st.slider("Retrieved chunks (k)", 2, 8, 4, key="topk")
    show_sources = st.checkbox("Show evidence chunks", value=True)

    st.divider()
    st.markdown('<p style="font-size:10px;color:var(--text3);text-align:center;font-family:var(--font-mono);">Gemini 2.0 Flash · MiniLM-L6-v2<br>LangChain 1.x · ChromaDB 1.x</p>',
                unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
active_meta = st.session_state.collections.get(st.session_state.active_col)

if not active_meta:
    st.markdown("""
    <div class="empty">
      <div class="empty-icon">🔬</div>
      <p class="empty-title">ScholarMind</p>
      <p class="empty-sub">Your AI-powered research paper analyst.<br>Upload a paper to unlock all 7 analysis modes.</p>
    </div>
    """, unsafe_allow_html=True)

    # Feature grid
    features = [
        ("💬", "Grounded Chat", "Ask anything — every answer cites evidence"),
        ("📋", "Auto-Summary", "Problem · Method · Results · Contribution"),
        ("🏷️", "Key Concepts", "Extract & deep-dive into technical terms"),
        ("🔍", "Research Gaps", "Find explicit + implicit gaps & future work"),
        ("📚", "Flashcards", "Auto-generate study cards from the paper"),
        ("⚖️", "Compare", "Side-by-side analysis of two papers"),
        ("🧒", "ELI15", "Explain paper to a 15-year-old"),
        ("📤", "Export", "Chat · Summary · JSON download"),
    ]
    cols = st.columns(4)
    for i, (icon, title, desc) in enumerate(features):
        with cols[i % 4]:
            st.markdown(f"""
            <div style="background:var(--bg2);border:1px solid var(--border);border-radius:var(--r-md);
                        padding:14px;margin-bottom:10px;text-align:center;">
              <div style="font-size:22px;margin-bottom:6px;">{icon}</div>
              <div style="font-size:12.5px;font-weight:600;color:var(--text0);margin-bottom:4px;">{title}</div>
              <div style="font-size:11px;color:var(--text2);line-height:1.5;">{desc}</div>
            </div>""", unsafe_allow_html=True)
    st.stop()

# ── Top bar ────────────────────────────────────────────────────────────────────
col_h1, col_h2, col_h3, col_h4 = st.columns([5, 1, 1, 1])
with col_h1:
    st.markdown(f"""
    <p style="font-size:11px;color:var(--text3);font-family:var(--font-mono);margin-bottom:1px;">ACTIVE PAPER</p>
    <p style="font-size:17px;font-weight:700;color:var(--text0);margin:0;letter-spacing:-.3px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{active_meta['filename']}</p>
    """, unsafe_allow_html=True)
with col_h2:
    st.metric("Chunks", active_meta["chunks"])
with col_h3:
    st.metric("Pages", active_meta.get("pages", "?"))
with col_h4:
    st.metric("k", top_k)

st.markdown('<hr style="margin:10px 0 0 !important;">', unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
t_chat, t_summary, t_concepts, t_gaps, t_flash, t_compare, t_export = st.tabs([
    "💬 Chat", "📋 Summary", "🏷️ Concepts",
    "🔍 Research Gaps", "📚 Flashcards", "⚖️ Compare", "📤 Export",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CHAT
# ══════════════════════════════════════════════════════════════════════════════
with t_chat:
    if not st.session_state.messages:
        chips = [
            "What is the main contribution?",
            "Summarize the methodology",
            "What are the key quantitative results?",
            "What datasets or benchmarks were used?",
            "What are the main limitations?",
            "How does this compare to prior work?",
        ]
        st.markdown('<div class="chip-grid">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        for i, q in enumerate(chips):
            if [c1, c2, c3][i % 3].button(q, key=f"chip_{i}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": q})
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # Render history
    for msg in st.session_state.messages:
        _render_msg(msg)

    # Input
    if not api_key:
        st.info("🔑 Enter your Gemini API key in the sidebar to start chatting.")
    else:
        question = st.chat_input("Ask anything about this paper…")
        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.spinner("Retrieving evidence · Generating answer…"):
                try:
                    res = _rag_query(question, active_meta["vectordb"], api_key, top_k)
                    fol = _get_follow_ups(question, res["answer"], api_key)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": res["answer"],
                        "sources": res["sources"] if show_sources else [],
                        "confidence": res["confidence"],
                        "follow_ups": fol,
                    })
                    st.session_state.follow_ups = fol
                except Exception as e:
                    st.session_state.messages.append({
                        "role": "assistant", "content": f"⚠️ Error: {e}",
                        "sources": [], "confidence": "low", "follow_ups": [],
                    })
            st.rerun()

    if st.session_state.messages:
        st.button("🗑 Clear chat", key="clear_chat",
                  on_click=lambda: st.session_state.update({"messages": [], "follow_ups": []}))

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
with t_summary:
    col_id = st.session_state.active_col
    if not api_key:
        st.info("🔑 Add your API key to generate a summary.")
    elif col_id not in st.session_state.summaries:
        st.markdown('<p style="color:var(--text1);font-size:13px;margin-bottom:12px;">Generate a structured analysis of the paper — problem, method, results, contribution, limitations.</p>', unsafe_allow_html=True)
        if st.button("📋 Analyse Paper", type="primary", key="gen_sum"):
            with st.spinner("Analysing paper with Gemini 2.0 Flash…"):
                try:
                    _get_summary(col_id, active_meta["vectordb"], api_key)
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed: {e}")
    else:
        s = st.session_state.summaries[col_id]

        # Hero card
        st.markdown(f"""
        <div class="hero-card">
          <div class="hero-eyebrow">paper in one line</div>
          <p class="hero-text">{s.get('one_liner','')}</p>
        </div>""", unsafe_allow_html=True)

        # Meta pills row
        ptype = s.get("paper_type", "").upper()
        venue = s.get("venue_guess", "?")
        year  = s.get("year_guess", "?")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Type", ptype)
        c2.metric("Venue", venue[:12])
        c3.metric("Year", year)
        c4.metric("Keywords", len(s.get("keywords", [])))

        st.markdown("<br>", unsafe_allow_html=True)

        # Sections
        sections = [
            ("problem",       "🎯 Problem"),
            ("method",        "⚙️ Methodology"),
            ("results",       "📊 Results"),
            ("contribution",  "💡 Contribution"),
            ("limitations",   "⚠️ Limitations"),
        ]
        for key, label in sections:
            text = s.get(key, "")
            if text:
                st.markdown(f"""
                <div class="info-card">
                  <div class="info-eyebrow">{label}</div>
                  <div class="info-text">{text}</div>
                </div>""", unsafe_allow_html=True)

        if st.button("🔄 Regenerate", key="regen_sum"):
            del st.session_state.summaries[col_id]
            st.rerun()

        st.download_button("⬇️ Download Summary (Markdown)",
                           data=_export_summary_md(s, active_meta["filename"]),
                           file_name=f"summary_{col_id}.md",
                           mime="text/markdown", key="dl_sum")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — KEY CONCEPTS
# ══════════════════════════════════════════════════════════════════════════════
with t_concepts:
    col_id = st.session_state.active_col
    if not api_key:
        st.info("🔑 Add your API key.")
    elif col_id not in st.session_state.summaries:
        if st.button("🏷️ Extract Concepts", type="primary", key="gen_conc"):
            with st.spinner("Extracting key concepts…"):
                try:
                    _get_summary(col_id, active_meta["vectordb"], api_key)
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed: {e}")
    else:
        s = st.session_state.summaries[col_id]
        keywords = s.get("keywords", [])
        if keywords:
            # Tag cloud
            tags = "".join(f'<span class="tag">{kw}</span>' for kw in keywords)
            st.markdown(f'<div style="margin-bottom:20px;">{tags}</div>', unsafe_allow_html=True)

            # Concept frequency chart using Plotly
            st.markdown('<div class="sec-head">Concept Relevance (retrieval frequency)</div>', unsafe_allow_html=True)
            cid = col_id
            freq_key = f"freq_{cid}"
            if freq_key not in st.session_state:
                with st.spinner("Scoring concept relevance…"):
                    emb = _load_embeddings()
                    scores = []
                    docs = active_meta["vectordb"].get()
                    all_texts = " ".join(docs.get("documents", [])[:20]).lower()
                    for kw in keywords:
                        count = all_texts.count(kw.lower())
                        scores.append(count if count > 0 else 1)
                    st.session_state[freq_key] = scores

            scores = st.session_state[freq_key]
            fig = go.Figure(go.Bar(
                x=scores, y=keywords, orientation="h",
                marker=dict(
                    color=scores,
                    colorscale=[[0, "#2D3A52"], [0.5, "#5B5FD9"], [1.0, "#7C83FD"]],
                    showscale=False,
                ),
                hovertemplate="%{y}: %{x} occurrences<extra></extra>",
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="JetBrains Mono, monospace", color="#A0ABCC", size=11),
                margin=dict(l=0, r=10, t=10, b=10),
                height=280,
                xaxis=dict(gridcolor="rgba(99,120,180,0.1)", zeroline=False, title="Occurrence count"),
                yaxis=dict(gridcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            # Deep-dive buttons
            st.markdown('<div class="sec-head">Deep-dive — ask about a concept</div>', unsafe_allow_html=True)
            if api_key:
                cols_kw = st.columns(3)
                for i, kw in enumerate(keywords[:9]):
                    if cols_kw[i % 3].button(f"⚡ {kw}", key=f"kw_{i}", use_container_width=True):
                        q = f"Explain how '{kw}' is used and why it matters in this paper."
                        st.session_state.messages.append({"role": "user", "content": q})
                        with st.spinner(f"Looking up '{kw}'…"):
                            try:
                                res = _rag_query(q, active_meta["vectordb"], api_key, top_k)
                                fol = _get_follow_ups(q, res["answer"], api_key)
                                st.session_state.messages.append({
                                    "role": "assistant", "content": res["answer"],
                                    "sources": res["sources"] if show_sources else [],
                                    "confidence": res["confidence"], "follow_ups": fol,
                                })
                            except Exception as e:
                                st.session_state.messages.append({
                                    "role": "assistant", "content": f"⚠️ {e}",
                                    "sources": [], "confidence": "low", "follow_ups": [],
                                })
                        st.info(f"Answer for '{kw}' added to the Chat tab ↑")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — RESEARCH GAPS
# ══════════════════════════════════════════════════════════════════════════════
with t_gaps:
    col_id = st.session_state.active_col
    gap_key = f"gaps_{col_id}"
    if not api_key:
        st.info("🔑 Add your API key.")
    elif gap_key not in st.session_state:
        st.markdown('<p style="color:var(--text1);font-size:13px;margin-bottom:12px;">Identify research gaps, limitations, and future directions this paper opens up.</p>', unsafe_allow_html=True)
        if st.button("🔍 Find Research Gaps", type="primary", key="gen_gaps"):
            with st.spinner("Analysing gaps and future directions…"):
                try:
                    _get_gaps(col_id, active_meta["vectordb"], api_key)
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed: {e}")
    else:
        gaps = st.session_state[gap_key]
        sections = [
            ("explicit_gaps",    "📌 Explicitly Acknowledged Gaps",     "tag"),
            ("implicit_gaps",    "🔎 Implicit Gaps (Not Addressed)",     "tag-amber"),
            ("future_directions","🚀 Future Research Directions",        "tag-green"),
            ("open_questions",   "❓ Open Questions This Paper Raises",  "tag"),
        ]
        for key, label, tag_cls in sections:
            items = gaps.get(key, [])
            if items:
                st.markdown(f'<div class="sec-head">{label}</div>', unsafe_allow_html=True)
                tags_html = "".join(f'<span class="tag {tag_cls}">{item}</span>' for item in items)
                st.markdown(f'<div style="margin-bottom:6px;">{tags_html}</div>', unsafe_allow_html=True)

                # Allow asking about each gap
                for j, item in enumerate(items[:3]):
                    if st.button(f"💬 Discuss: {item[:60]}…" if len(item) > 60 else f"💬 Discuss: {item}",
                                  key=f"gap_{key}_{j}", use_container_width=True):
                        q = f"Regarding the gap: '{item}' — what does the paper say, and what would addressing this require?"
                        st.session_state.messages.append({"role": "user", "content": q})
                        with st.spinner("Generating response…"):
                            try:
                                res = _rag_query(q, active_meta["vectordb"], api_key, top_k)
                                fol = _get_follow_ups(q, res["answer"], api_key)
                                st.session_state.messages.append({
                                    "role": "assistant", "content": res["answer"],
                                    "sources": res["sources"] if show_sources else [],
                                    "confidence": res["confidence"], "follow_ups": fol,
                                })
                            except Exception as e:
                                st.session_state.messages.append({
                                    "role": "assistant", "content": f"⚠️ {e}",
                                    "sources": [], "confidence": "low", "follow_ups": [],
                                })
                        st.info("Response added to Chat tab ↑")

        if st.button("🔄 Regenerate Gap Analysis", key="regen_gaps"):
            del st.session_state[gap_key]
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — FLASHCARDS
# ══════════════════════════════════════════════════════════════════════════════
with t_flash:
    col_id = st.session_state.active_col
    fc_key = f"fc_{col_id}"
    if not api_key:
        st.info("🔑 Add your API key.")
    elif fc_key not in st.session_state:
        st.markdown('<p style="color:var(--text1);font-size:13px;margin-bottom:12px;">Auto-generate study flashcards — great for quickly memorising key contributions, methods, and findings.</p>', unsafe_allow_html=True)
        if st.button("📚 Generate Flashcards", type="primary", key="gen_fc"):
            with st.spinner("Creating flashcards…"):
                try:
                    _get_flashcards(col_id, active_meta["vectordb"], api_key)
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed: {e}")
    else:
        cards = st.session_state[fc_key]
        if not cards:
            st.warning("Could not generate flashcards. Try regenerating.")
        else:
            eli5_key = f"eli5_{col_id}"

            c_left, c_right = st.columns([3, 1])
            with c_left:
                st.markdown(f'<div class="sec-head">{len(cards)} Flashcards</div>', unsafe_allow_html=True)
            with c_right:
                # ELI15 toggle
                if st.button("🧒 ELI15 Mode", key="eli5_btn", use_container_width=True):
                    if eli5_key not in st.session_state:
                        with st.spinner("Simplifying…"):
                            try:
                                _get_eli5(col_id, active_meta["vectordb"], api_key)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed: {e}")

            if eli5_key in st.session_state:
                st.markdown(f"""
                <div class="hero-card" style="margin-bottom:18px;">
                  <div class="hero-eyebrow">🧒 ELI15 — Explain Like I'm 15</div>
                  <p style="font-size:14px;color:var(--text1);line-height:1.8;margin:0;">{st.session_state[eli5_key]}</p>
                </div>""", unsafe_allow_html=True)

            # Render cards in 2 columns
            left_cards = cards[::2]
            right_cards = cards[1::2]
            fc_c1, fc_c2 = st.columns(2)
            for i, card in enumerate(left_cards):
                with fc_c1:
                    with st.expander(f"Card {i*2+1} — {card.get('q','')[:55]}…" if len(card.get('q','')) > 55 else f"Card {i*2+1} — {card.get('q','')}"):
                        st.markdown(f'<div style="font-size:13px;color:var(--text0);line-height:1.7;">{card.get("a","")}</div>', unsafe_allow_html=True)
            for i, card in enumerate(right_cards):
                with fc_c2:
                    with st.expander(f"Card {i*2+2} — {card.get('q','')[:55]}…" if len(card.get('q','')) > 55 else f"Card {i*2+2} — {card.get('q','')}"):
                        st.markdown(f'<div style="font-size:13px;color:var(--text0);line-height:1.7;">{card.get("a","")}</div>', unsafe_allow_html=True)

        if st.button("🔄 Regenerate Flashcards", key="regen_fc"):
            for k in [fc_key, f"eli5_{col_id}"]:
                st.session_state.pop(k, None)
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — COMPARE
# ══════════════════════════════════════════════════════════════════════════════
with t_compare:
    papers = list(st.session_state.collections.items())
    if len(papers) < 2:
        n = len(papers)
        st.markdown(f"""
        <div style="text-align:center;padding:50px 20px;">
          <div style="font-size:40px;margin-bottom:12px;">⚖️</div>
          <p style="font-size:15px;color:var(--text0);font-weight:600;">Upload at least 2 papers to compare</p>
          <p style="font-size:13px;color:var(--text1);">You have {n} paper{"" if n==1 else "s"} indexed. Upload another to unlock comparison.</p>
        </div>""", unsafe_allow_html=True)
    elif not api_key:
        st.info("🔑 Add your API key.")
    else:
        pmap = {cid: meta["filename"] for cid, meta in papers}
        sel_a = st.selectbox("Paper A", list(pmap.keys()), format_func=lambda x: pmap[x], key="cmp_a")
        opts_b = [k for k in pmap if k != sel_a]
        sel_b = st.selectbox("Paper B", opts_b, format_func=lambda x: pmap[x], key="cmp_b")

        if st.button("⚖️ Compare Papers", type="primary", use_container_width=True, key="run_cmp"):
            with st.spinner("Comparing papers with Gemini 2.0 Flash…"):
                try:
                    _get_compare(sel_a, sel_b, api_key)
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed: {e}")

        ckey = f"{sel_a}__{sel_b}"
        rkey = f"{sel_b}__{sel_a}"
        cdata = st.session_state.compare_cache.get(ckey) or st.session_state.compare_cache.get(rkey)
        if cdata:
            st.markdown("<br>", unsafe_allow_html=True)
            l, r = st.columns(2)
            with l:
                st.markdown(f"""
                <div class="cmp-card">
                  <div class="cmp-head">📄 {pmap[sel_a]}</div>
                  <div class="cmp-lbl">Contribution</div>
                  <div class="cmp-val">{cdata.get('a_contribution','')}</div>
                  <div class="cmp-lbl">Strength</div>
                  <div class="cmp-val">{cdata.get('a_strength','')}</div>
                </div>""", unsafe_allow_html=True)
            with r:
                st.markdown(f"""
                <div class="cmp-card">
                  <div class="cmp-head">📄 {pmap[sel_b]}</div>
                  <div class="cmp-lbl">Contribution</div>
                  <div class="cmp-val">{cdata.get('b_contribution','')}</div>
                  <div class="cmp-lbl">Strength</div>
                  <div class="cmp-val">{cdata.get('b_strength','')}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            sl, sr = st.columns(2)
            with sl:
                st.markdown("**🤝 Similarities**")
                for sim in cdata.get("similarities", []):
                    st.markdown(f"- {sim}")
            with sr:
                st.markdown("**🔀 Differences**")
                for diff in cdata.get("differences", []):
                    st.markdown(f"- {diff}")

            st.markdown(f"""
            <div class="info-card" style="margin-top:16px;">
              <div class="info-eyebrow">💡 Synthesis & Reading Order</div>
              <div class="info-text">{cdata.get('complementary','')}</div>
              <div style="margin-top:8px;font-size:12px;color:var(--accent);font-family:var(--font-mono);">📖 {cdata.get('read_order','')}</div>
            </div>""", unsafe_allow_html=True)

            # Comparison radar chart
            st.markdown('<div class="sec-head">Visual Comparison</div>', unsafe_allow_html=True)
            sim_count = len(cdata.get("similarities", []))
            diff_count = len(cdata.get("differences", []))
            fig2 = go.Figure()
            names = [pmap[sel_a][:20], pmap[sel_b][:20]]
            # Simple bar comparison of sim/diff counts
            fig2.add_trace(go.Bar(name="Shared aspects", x=names,
                                   y=[sim_count, sim_count],
                                   marker_color="#34D399", opacity=0.8))
            fig2.add_trace(go.Bar(name="Unique aspects", x=names,
                                   y=[diff_count // 2, diff_count - diff_count // 2],
                                   marker_color="#7C83FD", opacity=0.8))
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="JetBrains Mono, monospace", color="#A0ABCC", size=11),
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#A0ABCC")),
                margin=dict(l=0, r=10, t=10, b=20), height=200, barmode="group",
                xaxis=dict(gridcolor="rgba(99,120,180,0.1)"),
                yaxis=dict(gridcolor="rgba(99,120,180,0.1)", title="Count"),
            )
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — EXPORT
# ══════════════════════════════════════════════════════════════════════════════
with t_export:
    col_id = st.session_state.active_col
    fname = active_meta["filename"]
    msgs = st.session_state.messages

    st.markdown('<p style="font-size:13px;color:var(--text1);margin-bottom:18px;">Download your research session in multiple formats.</p>', unsafe_allow_html=True)

    e1, e2, e3 = st.columns(3)
    with e1:
        st.markdown(f"""
        <div style="background:var(--bg3);border:1px solid var(--border);border-radius:var(--r-md);padding:16px 16px 12px;">
          <div style="font-size:13px;font-weight:600;color:var(--text0);margin-bottom:4px;">💬 Chat Transcript</div>
          <div style="font-size:12px;color:var(--text1);line-height:1.6;margin-bottom:12px;">{len([m for m in msgs if m['role']=='user'])} questions · Full evidence chains</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if msgs:
            st.download_button("⬇️ Chat (.md)",
                               data=_export_md(msgs, fname),
                               file_name=f"chat_{col_id}.md",
                               mime="text/markdown", key="dl_chat",
                               use_container_width=True)
        else:
            st.caption("No messages yet.")

    with e2:
        st.markdown(f"""
        <div style="background:var(--bg3);border:1px solid var(--border);border-radius:var(--r-md);padding:16px 16px 12px;">
          <div style="font-size:13px;font-weight:600;color:var(--text0);margin-bottom:4px;">📋 Paper Summary</div>
          <div style="font-size:12px;color:var(--text1);line-height:1.6;margin-bottom:12px;">Structured analysis including keywords, type, venue</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if col_id in st.session_state.summaries:
            s = st.session_state.summaries[col_id]
            st.download_button("⬇️ Summary (.md)",
                               data=_export_summary_md(s, fname),
                               file_name=f"summary_{col_id}.md",
                               mime="text/markdown", key="dl_sum_exp",
                               use_container_width=True)
        else:
            st.caption("Generate summary first.")

    with e3:
        st.markdown(f"""
        <div style="background:var(--bg3);border:1px solid var(--border);border-radius:var(--r-md);padding:16px 16px 12px;">
          <div style="font-size:13px;font-weight:600;color:var(--text0);margin-bottom:4px;">🗂 Full Session (JSON)</div>
          <div style="font-size:12px;color:var(--text1);line-height:1.6;margin-bottom:12px;">Messages · Summary · Gaps · Flashcards</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        session = {
            "exported_at": datetime.datetime.now().isoformat(),
            "paper": fname, "chunks": active_meta["chunks"],
            "messages": [{"role": m["role"], "content": m["content"],
                          "confidence": m.get("confidence"),
                          "sources": m.get("sources", [])} for m in msgs],
            "summary": st.session_state.summaries.get(col_id, {}),
            "gaps": st.session_state.get(f"gaps_{col_id}", {}),
            "flashcards": st.session_state.get(f"fc_{col_id}", []),
        }
        st.download_button("⬇️ Session (.json)",
                           data=json.dumps(session, indent=2),
                           file_name=f"session_{col_id}.json",
                           mime="application/json", key="dl_json",
                           use_container_width=True)

    # Session stats
    st.markdown('<div class="sec-head">Session Stats</div>', unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Questions asked", len([m for m in msgs if m["role"] == "user"]))
    s2.metric("Summary ready", "Yes" if col_id in st.session_state.summaries else "No")
    s3.metric("Gaps analysed", "Yes" if f"gaps_{col_id}" in st.session_state else "No")
    s4.metric("Flashcards", len(st.session_state.get(f"fc_{col_id}", [])))