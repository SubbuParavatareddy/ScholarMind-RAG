# 🔬 ScholarMind — AI Research Paper Assistant

> ** for research papers. Powered by **Gemini 2.5 Flash**, **LangChain 1.x**, and **ChromaDB 1.x** — deploy free on Streamlit Community Cloud.

**Live demo:** `https://your-app.streamlit.app`

---

## ✨ Feature Set (8 Modes)

| Tab | Feature | What it does |
|---|---|---|
| 💬 | **Grounded Chat** | MMR RAG with evidence chunks, confidence scores, auto follow-up suggestions |
| 📋 | **Auto-Summary** | Structured: problem · method · results · contribution · limitations · venue/year |
| 🏷️ | **Key Concepts** | Keywords + Plotly frequency chart + one-click deep-dive queries |
| 🔍 | **Research Gaps** | Explicit gaps · implicit gaps · future directions · open questions |
| 📚 | **Flashcards** | 6 auto-generated study cards + ELI15 (explain like I'm 15) mode |
| ⚖️ | **Compare** | Side-by-side comparison of 2 papers with visual chart |
| 📤 | **Export** | Chat .md · Summary .md · Full session .json |

---

## 🚀 Deploy Free (3 Steps)

```bash
# 1. Push to public GitHub repo
git init && git add . && git commit -m "init" && git push

# 2. Go to share.streamlit.io → New app → select repo → app.py → Deploy

# 3. (Optional) Add secret: Settings → Secrets → GEMINI_API_KEY = "..."
```

## 💻 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Get a free key at [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey).

---

## 🔧 Stack

| Component | Package          | Version |
|---|------------------|---|
| Framework | streamlit        | 1.41.0 |
| LLM | gemini-2.5-flash | via langchain-google-genai 4.2.5 |
| Embeddings | all-MiniLM-L6-v2 | sentence-transformers 3.4.1 |
| Vector DB | ChromaDB         | 1.0.21 |
| RAG | LangChain        | 1.3.6 |
| PDF | pypdf            | 6.7.5 |
| Charts | Plotly           | 6.1.0 |

## 🆓 Free Hosting Options

| Platform | Free RAM | Notes |
|---|---|---|
| **⭐ Streamlit Community Cloud** | ~1 GB | Purpose-built, `share.streamlit.io`, 100% free for public repos |
| **Hugging Face Spaces** | 16 GB | Best for large models |
| **Railway** | 512 MB | $5/mo free credit |
| **Render** | 512 MB | Spins down after inactivity |
