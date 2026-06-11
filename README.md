# ScholarMind — AI Research Paper Assistant

> Conversational RAG over your research papers. Powered by **Gemini 2.5 Flash**, **LangChain 0.3 (LCEL)**, and a **pure-NumPy vector store** — deploy free on Streamlit Community Cloud.

**Live demo:** `https://your-app.streamlit.app`

---

## Features (7 Tabs)

| Tab | What it does |
|---|---|
| 💬 **Chat** | MMR-grounded RAG with confidence badges, source chunks, and 3 AI-generated follow-up buttons |
| 📋 **Summary** | Structured summary: problem · method · results · contribution · limitations · venue/year |
| 🏷️ **Concepts** | Keyword extraction + Plotly frequency chart + one-click deep-dive into Chat |
| 🔍 **Research Gaps** | Explicit gaps · implicit gaps · future directions · open questions + Discuss in Chat |
| 📚 **Flashcards** | 6 auto-generated Q&A cards + ELI15 (Explain Like I'm 15) mode |
| ⚖️ **Compare** | Side-by-side LLM comparison of two loaded papers with a visual chart |
| 📤 **Export** | Chat as `.md` · Summary as `.md` · Full session as `.json` |

---

## Stack

| Component | Package | Version |
|---|---|---|
| Framework | streamlit | 1.41.0 |
| LLM | gemini-2.5-flash | via langchain-google-genai 2.1.4 |
| Embeddings | all-MiniLM-L6-v2 | sentence-transformers 3.4.1 |
| Vector store | NumpyVectorStore (custom) | numpy ≥ 1.26.0 |
| RAG | LangChain LCEL | langchain 0.3.27 |
| PDF parsing | pypdf | 6.7.5 |
| Charts | plotly | 6.1.0 |

> **No ChromaDB, no SQLite, no protobuf.** The custom `NumpyVectorStore` handles cosine similarity and MMR retrieval in pure NumPy — eliminating all dependency conflicts on Python 3.12.

---

## Project Structure

```
ScholarMind-RAG/
├── app.py                  # Streamlit UI entry point (thin layer)
├── backend/
│   ├── config.py           # APIKeyLoader (st.secrets → config.toml fallback)
│   ├── vector_store.py     # NumpyVectorStore: cosine similarity + MMR
│   ├── document_loader.py  # PDF/TXT/MD → chunks → embeddings → NumpyVectorStore
│   ├── rag_engine.py       # LCEL RAG chain + follow-up generation
│   ├── analysis.py         # PaperAnalyzer: summary, flashcards, gaps, ELI15, compare
│   └── exporters.py        # Markdown / JSON exporters
├── frontend/
│   └── styles.css          # Custom dark-theme CSS (injected via st.markdown)
├── .streamlit/
│   └── config.toml         # Theme + server config + local [secrets] (gitignored)
├── docs/
│   └── REPORT.md           # Full technical report
├── requirements.txt
└── runtime.txt             # python-3.12
```

---

## Deploy Free on Streamlit Community Cloud

```bash
# 1. Push to a public GitHub repo
git push origin master

# 2. Go to share.streamlit.io → New app → select repo → app.py → Deploy

# 3. Add your API key: App Settings → Secrets
#    GOOGLE_API_KEY = "your-key-here"
```

Get a free key at [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey).

---

## Run Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add your API key to .streamlit/config.toml
#    (create the file if it doesn't exist — it is gitignored)
```

`.streamlit/config.toml`:
```toml
[theme]
base = "dark"

[server]
maxUploadSize = 50

[secrets]
GOOGLE_API_KEY = "your-key-here"
```

```bash
# 3. Run
streamlit run app.py
```

---

## API Key Resolution

`APIKeyLoader` checks two locations in order:

1. `st.secrets["GOOGLE_API_KEY"]` — set by Streamlit Cloud / GitHub Secrets
2. `.streamlit/config.toml` under `[secrets]` — local development

The sidebar shows a ✓ or ⚠ status badge. There is no text-input widget for the key.

---

## Free Hosting Options

| Platform | Free RAM | Notes |
|---|---|---|
| **Streamlit Community Cloud** | ~1 GB | Purpose-built; `share.streamlit.io`; 100% free for public repos |
| **Hugging Face Spaces** | 16 GB | Better for GPU or large models |
| **Railway** | 512 MB | $5/mo free credit |
| **Render** | 512 MB | Spins down after inactivity |
