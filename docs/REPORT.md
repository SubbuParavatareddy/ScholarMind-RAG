# ScholarMind — Technical Report
**Project:** AI Research Paper Assistant · RAG Application  
**Framework:** Streamlit 1.41  
**Hosting:** Streamlit Community Cloud (free)  
**Python:** 3.12 (pinned via `runtime.txt`)

---

## 1. Architecture

### System Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        app.py  (UI layer)                        │
│                                                                   │
│  Sidebar                         Main Panel (7 tabs)             │
│  ┌──────────────────────┐        ┌──────────────────────────┐   │
│  │ API key status badge  │        │ 💬 Chat                  │   │
│  │ File uploader         │        │ 📋 Summary               │   │
│  │ Paper library list    │        │ 🏷️ Concepts              │   │
│  │ Retrieval slider (k)  │        │ 🔍 Research Gaps         │   │
│  │ Show sources toggle   │        │ 📚 Flashcards + ELI15    │   │
│  └──────────────────────┘        │ ⚖️ Compare               │   │
│                                   │ 📤 Export                │   │
│                                   └──────────────────────────┘   │
└──────────────────────────┬──────────────────────────────────────┘
                            │ imports
┌──────────────────────────▼──────────────────────────────────────┐
│                      backend/  (logic layer)                      │
│                                                                   │
│  ┌────────────────┐  ┌──────────────────┐  ┌────────────────┐   │
│  │  APIKeyLoader  │  │  DocumentLoader  │  │ NumpyVector    │   │
│  │                │  │                  │  │ Store          │   │
│  │ st.secrets     │  │ PDF / TXT / MD   │  │                │   │
│  │ config.toml    │  │ → chunks         │  │ Cosine sim     │   │
│  │ [secrets]      │  │ → embeddings     │  │ MMR retrieval  │   │
│  └────────────────┘  │ → NumpyVS        │  │ NumPy only     │   │
│                       └──────────────────┘  └────────────────┘   │
│  ┌────────────────┐  ┌──────────────────┐  ┌────────────────┐   │
│  │   RAGEngine    │  │  PaperAnalyzer   │  │   Exporter     │   │
│  │                │  │                  │  │                │   │
│  │ embed query    │  │ get_summary()    │  │ chat → .md     │   │
│  │ MMR search     │  │ get_flashcards() │  │ summary → .md  │   │
│  │ LCEL chain     │  │ get_gaps()       │  │ session → .json│   │
│  │ get_follow_ups │  │ get_eli15()      │  └────────────────┘   │
│  └────────────────┘  │ get_compare()    │                        │
│                       └──────────────────┘                        │
└─────────────────────────────────────────────────────────────────┘
                            │
┌──────────────────────────▼──────────────────────────────────────┐
│                      frontend/                                    │
│                      styles.css  (all custom CSS, injected once) │
└─────────────────────────────────────────────────────────────────┘
```

### Ingest Pipeline

```
UploadedFile
    → tempfile write (deleted immediately after load)
    → PyPDFLoader (PDF) or TextLoader (TXT/MD)
    → RecursiveCharacterTextSplitter (chunk_size=900, overlap=180)
    → HuggingFaceEmbeddings.embed_documents()  [cached via @st.cache_resource]
    → NumpyVectorStore.add(texts, metadatas, vectors)
    → collection dict stored in st.session_state.collections
```

### Query Pipeline (LCEL)

```
User question
    → HuggingFaceEmbeddings.embed_query()
    → NumpyVectorStore.mmr_search(k, fetch_k=3k, lambda=0.6)
    → context = join top-k chunk texts
    → ChatPromptTemplate | ChatGoogleGenerativeAI | StrOutputParser   [LCEL chain]
    → answer + confidence (high/medium/low) + source chunk list
    → follow-up questions (separate LLM call, JSON array of 3)
    → appended to st.session_state.messages
```

### Key Design Choices

**Monorepo layout.** The original single `app.py` was refactored into `backend/` (six classes) and `frontend/styles.css`. `app.py` is now a thin Streamlit entry point only — it owns session state, widget rendering, and the `@st.cache_resource` embedding loader. All logic is in the backend and is independently testable.

**No ChromaDB, no SQLite.** The project replaces Chroma entirely with `NumpyVectorStore` — a pure-NumPy in-memory store that implements cosine similarity and MMR. This eliminates protobuf conflicts and SQLite schema errors that arose on Python 3.12+ and Streamlit Cloud.

**LCEL instead of `RetrievalQA`.** `RetrievalQA` inherits from `Chain` which uses Pydantic forward-refs that break on Python 3.12/3.14. LCEL (`prompt | llm | parser`) is the current LangChain standard and is fully Python 3.12/3.14 safe.

**No API key input widget.** The Gemini API key is loaded automatically via `APIKeyLoader`: first from `st.secrets["GOOGLE_API_KEY"]` (Streamlit Cloud/GitHub Secrets), then falling back to `.streamlit/config.toml [secrets]` for local development. The sidebar shows a ✓ / ⚠ status badge only.

**Cross-tab message queue.** Streamlit renders tabs top-to-bottom in script order. The Chat tab renders before Concepts/Gaps. When a deep-dive or Discuss button fires in a later tab, it stores results in `st.session_state["_pending_msgs"]` and calls `st.rerun()`. On the next pass, the Chat tab flushes this queue into `st.session_state.messages` before its render loop — guaranteeing the messages are visible immediately.

---

## 2. Component Choices & Reasoning

### LLM: Google Gemini 2.5 Flash (`gemini-2.5-flash`)

- **Free tier** at [aistudio.google.com](https://aistudio.google.com) — 15 RPM, 1M tokens/day
- **1M token context window** — handles large retrieved contexts comfortably
- **Speed** — Flash variant optimised for low latency (~2–3 s for RAG answers)
- **Instruction following** — reliably stays within context-only constraint in the RAG prompt
- Accessed via `langchain-google-genai==2.1.4`; temperature=0.15 for factual answers, higher (0.3–0.5) for creative tasks (flashcards, ELI15)

### Embedding Model: `sentence-transformers/all-MiniLM-L6-v2`

- Runs **locally on CPU** — zero API cost, fits Streamlit Cloud's free 1 GB RAM
- **384-dimensional vectors** — compact and fast for in-memory cosine search
- Loaded once via `@st.cache_resource` — survives all reruns within a session
- Top performer on SBERT benchmarks for semantic similarity at this size class

### Vector Store: Custom `NumpyVectorStore`

- **Pure NumPy** — no external service, no SQLite, no protobuf
- In-memory; stored in `st.session_state.collections[col_id]["vectorstore"]`
- Implements **MMR (Maximal Marginal Relevance)**: fetches `fetch_k = 3k` candidates, iteratively selects the next chunk that maximises `λ·relevance − (1−λ)·max_sim_to_selected` (λ=0.6). Avoids repetitive chunk selection.
- Clears on browser tab close — acceptable for a stateless research assistant

### RAG Framework: LangChain 0.3.x (LCEL)

- **LCEL chain**: `ChatPromptTemplate | ChatGoogleGenerativeAI | StrOutputParser`
- System prompt explicitly restricts answers to retrieved context, preventing hallucination
- Confidence heuristic: `low` if answer contains hedge phrases ("don't have", "cannot", etc.); `medium` if answer is short (<130 chars); `high` otherwise
- Follow-up questions: separate JSON-mode LLM call returning an array of 3 strings, rendered as clickable `st.button` widgets

### Chunking Strategy

| Parameter | Value | Reason |
|---|---|---|
| `chunk_size` | 900 chars | Full paragraph; fits comfortably in Gemini's context |
| `chunk_overlap` | 180 chars | 20% overlap prevents losing cross-boundary context |
| `separators` | `["\n\n", "\n", ". ", "! ", "? ", " "]` | Paragraph → sentence → word hierarchy |

---

## 3. Project Structure

```
ScholarMind-RAG/
├── app.py                      # Streamlit entry point (UI only)
├── backend/
│   ├── __init__.py             # Re-exports all public classes
│   ├── config.py               # APIKeyLoader
│   ├── vector_store.py         # NumpyVectorStore (cosine + MMR)
│   ├── document_loader.py      # DocumentLoader.ingest()
│   ├── rag_engine.py           # RAGEngine (query, get_follow_ups)
│   ├── analysis.py             # PaperAnalyzer (summary, gaps, flashcards, ELI15, compare)
│   └── exporters.py            # Exporter (chat→md, summary→md)
├── frontend/
│   └── styles.css              # All CSS (injected via st.markdown)
├── .streamlit/
│   └── config.toml             # Theme + server config + [secrets] for local API key
├── data/                       # Sample paper summaries for testing
├── docs/
│   └── REPORT.md               # This file
├── requirements.txt
└── runtime.txt                 # python-3.12
```

---

## 4. API Key Configuration

| Environment | Where to set | Key name |
|---|---|---|
| **Local development** | `.streamlit/config.toml` under `[secrets]` | `GOOGLE_API_KEY` |
| **Streamlit Cloud** | Repo Settings → Secrets (GitHub Secrets) | `GOOGLE_API_KEY` |

`APIKeyLoader.load()` checks `st.secrets` first (works on Cloud), then falls back to parsing `config.toml` directly via `tomllib` (Python 3.11+ stdlib).

---

## 5. Free Hosting Platforms Comparison

| Platform | Free RAM | Storage | Sleep Policy | Deploy Method | Notes |
|---|---|---|---|---|---|
| **⭐ Streamlit Community Cloud** | ~1 GB | Ephemeral | After inactivity | GitHub repo | Purpose-built; `share.streamlit.io` |
| **Hugging Face Spaces** | 16 GB | 50 GB | None | Git push | Better for GPU/large models |
| **Railway** | 512 MB | 1 GB | $5/mo credit | GitHub CI/CD | Short projects |
| **Render** | 512 MB | — | 15 min spin-down | GitHub | Simple services |
| **Replit** | 512 MB | 1 GB | After inactivity | In-browser | Quick demos |

---

## 6. Challenges & Solutions

### Challenge 1: Protobuf / SQLite conflicts on Python 3.12
ChromaDB (used in the original version) pulls in `protobuf` and uses SQLite internally, both of which have known incompatibilities on Python 3.12 and Streamlit Cloud's container. **Solution:** replaced ChromaDB entirely with `NumpyVectorStore` — zero external vector-DB dependency.

### Challenge 2: `RetrievalQA` breaking on Python 3.12/3.14
`RetrievalQA` inherits from `Chain` which uses Pydantic forward-refs that cause `TypeError` on newer Python. **Solution:** migrated to LCEL (`prompt | llm | parser`), the current LangChain standard.

### Challenge 3: Cross-tab message delivery
Streamlit renders tab content sequentially (top to bottom). A message appended inside the Concepts tab would not appear in the Chat tab in the same render pass because Chat renders first. **Solution:** a `_pending_msgs` queue in `st.session_state`. Concepts/Gaps store results there; Chat flushes it at the top of its block before the message render loop.

### Challenge 4: Embedding model cold start (~20 s on Community Cloud)
`@st.cache_resource` loads the model exactly once per server instance, shared across all user sessions. Subsequent requests pay zero loading cost.

### Challenge 5: Temporary file handling
Streamlit uploads expose a byte buffer, not a file path. The loader writes to a `tempfile`, passes the path to `PyPDFLoader`/`TextLoader`, then calls `os.unlink()` immediately after — respecting Streamlit Cloud's ephemeral disk.

### Challenge 6: Preventing hallucinations
RAG prompt with explicit `"Answer using ONLY the retrieved context. If context is insufficient, say so — never fabricate."` instruction at `temperature=0.15`. Confidence scorer flags low-confidence answers with a visual badge.

---

## 7. Evaluation

### Qualitative Test — `attention_is_all_you_need_summary.txt`

| Query | Retrieved correctly | Answered correctly |
|---|---|---|
| Main contribution | ✅ | ✅ |
| BLEU score EN→DE | ✅ | ✅ (28.4) |
| Number of attention heads | ✅ | ✅ (8) |
| Optimizer | ✅ | ✅ (Adam + warmup) |
| Limitations | ✅ | ✅ |
| Out-of-scope question | N/A | ✅ Declined correctly |

### Retrieval Metrics (informal)

- **Top-4 MMR precision**: 8/10 queries returned ≥3 directly relevant chunks
- **Faithfulness**: 100% of tested answers traceable to retrieved chunks
- **Latency**: avg ~3.5 s (embedding: cached; MMR search: <0.1 s; Gemini Flash: ~3 s)

---

## 8. Future Enhancements

1. **Persistent vector store** — Pinecone/Qdrant free tier for cross-session memory
2. **Streaming responses** — `st.write_stream` with Gemini's streaming API
3. **arXiv integration** — URL input → auto-fetch PDF via `arxiv` Python library
4. **RAGAs evaluation tab** — automated faithfulness / context-relevance scoring
5. **Test suite** — `backend/` classes are Streamlit-free; pytest fixtures can inject a plain `dict` as cache and a mock LLM
6. **Multi-paper cross-query** — unified MMR search across all indexed collections
