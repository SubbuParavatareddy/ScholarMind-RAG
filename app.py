"""
ScholarMind — AI Research Paper Assistant
Gemini 2.5 Flash · LangChain 0.3.x LCEL · Pure NumPy VectorStore · Streamlit 1.41
"""

import json
import datetime
from pathlib import Path

import streamlit as st

from backend import DocumentLoader, RAGEngine, PaperAnalyzer, Exporter, APIKeyLoader

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ScholarMind · Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
_css = (Path(__file__).parent / "frontend" / "styles.css").read_text()
st.markdown(f"<style>{_css}</style>", unsafe_allow_html=True)

# ── API key (no manual entry — loaded from config/secrets) ────────────────────
api_key = APIKeyLoader.load()

# ── Session state ──────────────────────────────────────────────────────────────
for _k, _v in {
    "messages": [],
    "collections": {},
    "active_col": None,
    "summaries": {},
    "compare_cache": {},
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── Embedding model — cached across reruns (Streamlit-layer concern) ──────────
@st.cache_resource(show_spinner="Loading embedding model…")
def _load_emb():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ── UI render helpers ──────────────────────────────────────────────────────────
def _conf_badge(c: str) -> str:
    m = {"high": ("● HIGH", "c-hi"), "medium": ("◑ MED", "c-md"), "low": ("○ LOW", "c-lo")}
    lbl, cls = m.get(c, ("? UNK", "c-lo"))
    return f'<span class="conf {cls}">{lbl}</span>'


def _render_msg(msg: dict):
    if msg["role"] == "user":
        st.markdown(
            f'<div class="row-u"><div class="bub-u">{msg["content"]}</div>'
            f'<div class="av av-u">👤</div></div>',
            unsafe_allow_html=True,
        )
    else:
        ans = msg["content"].replace("\n", "<br>")
        srcs = ""
        if msg.get("sources"):
            items = "".join(
                f'<div class="src-item"><span class="src-n">{i+1}</span>'
                f'<span class="src-pg">p.{s["page"]}</span>{s["content"]}</div>'
                for i, s in enumerate(msg["sources"])
            )
            srcs = (
                f'<div class="src-wrap"><div class="src-title">📎 Evidence · '
                f'{len(msg["sources"])} chunks</div>{items}</div>'
            )
        fols = ""
        if msg.get("follow_ups"):
            chips = "".join(f'<span class="fol-chip">{q}</span>' for q in msg["follow_ups"])
            fols = (
                f'<div style="margin-top:10px;padding-top:8px;border-top:1px solid var(--border)">'
                f'<div style="font-size:10px;color:var(--text2);font-family:var(--mono);margin-bottom:5px;">💡 FOLLOW-UPS</div>'
                f'{chips}</div>'
            )
        st.markdown(
            f'<div class="row-a"><div class="av av-a">🔬</div>'
            f'<div class="bub-a">{ans}{_conf_badge(msg.get("confidence","medium"))}{srcs}{fols}</div></div>',
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        '<div class="logo-wrap">'
        '<p class="logo-name">Scholar<em>Mind</em></p>'
        '<p class="logo-tag">Gemini 2.5 · NumPy RAG · Python 3.12</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    if api_key:
        st.markdown(
            '<p style="font-size:11px;color:#34D399;margin:4px 0 8px;">✓ API key loaded</p>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<p style="font-size:11px;color:#FBBF24;line-height:1.6;margin:4px 0 8px;">'
            '⚡ GOOGLE_API_KEY not set.<br>'
            '<b>Local:</b> add to .streamlit/config.toml [secrets].<br>'
            '<b>Deployed:</b> set GitHub Secret GOOGLE_API_KEY.</p>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown('<span class="sb-label">📄 Upload Paper</span>', unsafe_allow_html=True)
    uploaded = st.file_uploader("up", type=["pdf", "txt", "md"], label_visibility="collapsed")

    if uploaded and api_key:
        existing = {v["filename"] for v in st.session_state.collections.values()}
        if uploaded.name not in existing:
            with st.spinner("Indexing…"):
                try:
                    res = DocumentLoader.ingest(uploaded, _load_emb())
                    st.session_state.collections[res["col_id"]] = res
                    st.session_state.active_col = res["col_id"]
                    st.session_state.messages = []
                    st.success(f"✓ {res['chunks']} chunks indexed")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed: {e}")
        else:
            cid = next(k for k, v in st.session_state.collections.items()
                       if v["filename"] == uploaded.name)
            if st.session_state.active_col != cid:
                st.session_state.active_col = cid
                st.session_state.messages = []
                st.rerun()
    elif uploaded and not api_key:
        st.warning("Configure API key first.")

    st.divider()
    st.markdown('<span class="sb-label">📚 Paper Library</span>', unsafe_allow_html=True)
    if not st.session_state.collections:
        st.markdown(
            '<p style="font-size:12px;color:var(--text3);text-align:center;padding:8px 0;">No papers yet</p>',
            unsafe_allow_html=True,
        )
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
            st.markdown(
                f'<p style="font-size:10px;color:var(--text3);font-family:var(--mono);'
                f'margin:-4px 0 6px 2px;">{meta["chunks"]} chunks · {meta.get("pages","?")}p · {cid}</p>',
                unsafe_allow_html=True,
            )

    st.divider()
    st.markdown('<span class="sb-label">⚙️ Retrieval</span>', unsafe_allow_html=True)
    top_k = st.slider("Chunks (k)", 2, 8, 4)
    show_src = st.checkbox("Show evidence chunks", value=True)
    st.divider()
    st.markdown(
        '<p style="font-size:10px;color:var(--text3);text-align:center;font-family:var(--mono);">'
        'Gemini 2.5 Flash · MiniLM-L6-v2<br>NumPy MMR · No ChromaDB</p>',
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
active_meta = st.session_state.collections.get(st.session_state.active_col)

if not active_meta:
    st.markdown(
        '<div style="display:flex;flex-direction:column;align-items:center;'
        'justify-content:center;padding:60px 40px;text-align:center;">'
        '<div style="font-size:52px;margin-bottom:18px;">🔬</div>'
        '<p style="font-size:20px;font-weight:600;color:#EEF0F8;margin-bottom:10px;">ScholarMind</p>'
        '<p style="font-size:13.5px;color:#A0ABCC;line-height:1.75;max-width:420px;">'
        'Upload a research paper in the sidebar to unlock 7 analysis modes.</p></div>',
        unsafe_allow_html=True,
    )
    _cols = st.columns(4)
    for _i, (_icon, _title, _desc) in enumerate([
        ("💬", "Grounded Chat",  "MMR RAG + confidence scores"),
        ("📋", "Auto-Summary",   "Problem · Method · Results"),
        ("🔍", "Research Gaps",  "Explicit + implicit gaps"),
        ("📚", "Flashcards",     "Study cards + ELI15 mode"),
        ("🏷️", "Concepts",       "Keywords + frequency chart"),
        ("⚖️", "Compare",        "Side-by-side analysis"),
        ("💡", "Follow-ups",     "Auto next questions"),
        ("📤", "Export",         "MD + JSON downloads"),
    ]):
        with _cols[_i % 4]:
            st.markdown(
                f'<div style="background:var(--bg2);border:1px solid var(--border);'
                f'border-radius:var(--r-md);padding:14px;margin-bottom:10px;text-align:center;">'
                f'<div style="font-size:22px;margin-bottom:6px;">{_icon}</div>'
                f'<div style="font-size:12.5px;font-weight:600;color:var(--text0);margin-bottom:4px;">{_title}</div>'
                f'<div style="font-size:11px;color:var(--text2);line-height:1.5;">{_desc}</div></div>',
                unsafe_allow_html=True,
            )
    st.stop()

# ── Active paper header ────────────────────────────────────────────────────────
h1, h2, h3, h4 = st.columns([5, 1, 1, 1])
with h1:
    st.markdown(
        f'<p style="font-size:11px;color:var(--text3);font-family:var(--mono);margin-bottom:1px;">ACTIVE PAPER</p>'
        f'<p style="font-size:17px;font-weight:700;color:var(--text0);margin:0;overflow:hidden;'
        f'text-overflow:ellipsis;white-space:nowrap;">{active_meta["filename"]}</p>',
        unsafe_allow_html=True,
    )
h2.metric("Chunks", active_meta["chunks"])
h3.metric("Pages", active_meta.get("pages", "?"))
h4.metric("k", top_k)
st.markdown('<hr style="margin:10px 0 0!important;">', unsafe_allow_html=True)

# ── Backend instances (created per render; _load_emb is cached) ───────────────
_engine: RAGEngine | None = RAGEngine(api_key, _load_emb()) if api_key else None
_analyzer: PaperAnalyzer | None = PaperAnalyzer(_engine, st.session_state) if _engine else None

t_chat, t_sum, t_conc, t_gaps, t_flash, t_cmp, t_exp = st.tabs([
    "💬 Chat", "📋 Summary", "🏷️ Concepts", "🔍 Gaps",
    "📚 Flashcards", "⚖️ Compare", "📤 Export",
])

# ── CHAT ───────────────────────────────────────────────────────────────────────
with t_chat:
    for msg in st.session_state.messages:
        _render_msg(msg)

    if not api_key:
        st.info("🔑 Configure GOOGLE_API_KEY to enable chat (see sidebar).")
    else:
        question = st.chat_input("Ask anything about this paper…")
        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.spinner("Retrieving evidence · Generating answer…"):
                try:
                    res = _engine.query(question, active_meta["vectorstore"], top_k)
                    fol = _engine.get_follow_ups(question, res["answer"])
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

# ── SUMMARY ────────────────────────────────────────────────────────────────────
with t_sum:
    col_id = st.session_state.active_col
    if not api_key:
        st.info("🔑 Configure API key.")
    elif col_id not in st.session_state.summaries:
        if st.button("📋 Analyse Paper", type="primary", key="gen_sum"):
            with st.spinner("Analysing with Gemini 2.5 Flash…"):
                _analyzer.get_summary(col_id, active_meta["vectorstore"])
                st.rerun()
    else:
        s = st.session_state.summaries[col_id]
        st.markdown(
            f'<div class="hero-card"><div class="hero-eyebrow">paper in one line</div>'
            f'<p class="hero-text">{s.get("one_liner","")}</p></div>',
            unsafe_allow_html=True,
        )
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Type", s.get("paper_type", "?").upper()[:10])
        m2.metric("Venue", s.get("venue_guess", "?")[:12])
        m3.metric("Year", s.get("year_guess", "?"))
        m4.metric("Keywords", len(s.get("keywords", [])))
        st.markdown("<br>", unsafe_allow_html=True)
        for _key, _label in [
            ("problem", "🎯 Problem"), ("method", "⚙️ Methodology"),
            ("results", "📊 Results"), ("contribution", "💡 Contribution"),
            ("limitations", "⚠️ Limitations"),
        ]:
            if s.get(_key):
                st.markdown(
                    f'<div class="info-card"><div class="info-eyebrow">{_label}</div>'
                    f'<div class="info-text">{s[_key]}</div></div>',
                    unsafe_allow_html=True,
                )
        c1, c2 = st.columns(2)
        if c1.button("🔄 Regenerate", key="regen_sum"):
            st.session_state.summaries.pop(col_id, None)
            st.rerun()
        c2.download_button(
            "⬇️ Summary (.md)",
            data=Exporter.summary_to_markdown(s, active_meta["filename"]),
            file_name=f"summary_{col_id}.md", mime="text/markdown", key="dl_sum",
        )

# ── CONCEPTS ───────────────────────────────────────────────────────────────────
with t_conc:
    col_id = st.session_state.active_col
    if not api_key:
        st.info("🔑 Configure API key.")
    elif col_id not in st.session_state.summaries:
        if st.button("🏷️ Extract Concepts", type="primary", key="gen_conc"):
            with st.spinner("Extracting…"):
                _analyzer.get_summary(col_id, active_meta["vectorstore"])
                st.rerun()
    else:
        import plotly.graph_objects as go
        kws = st.session_state.summaries[col_id].get("keywords", [])
        if kws:
            st.markdown("".join(f'<span class="tag">{k}</span>' for k in kws) + "<br><br>",
                        unsafe_allow_html=True)
            all_texts = " ".join(active_meta["vectorstore"].get_all_texts()[:20]).lower()
            scores = [max(all_texts.count(k.lower()), 1) for k in kws]
            fig = go.Figure(go.Bar(
                x=scores, y=kws, orientation="h",
                marker=dict(color=scores,
                            colorscale=[[0, "#2D3A52"], [0.5, "#5B5FD9"], [1, "#7C83FD"]],
                            showscale=False),
                hovertemplate="%{y}: %{x}<extra></extra>",
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="JetBrains Mono", color="#A0ABCC", size=11),
                margin=dict(l=0, r=10, t=10, b=10), height=260,
                xaxis=dict(gridcolor="rgba(99,120,180,0.1)", zeroline=False),
                yaxis=dict(gridcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.markdown('<div class="sec-head">Deep-dive</div>', unsafe_allow_html=True)
            dcols = st.columns(3)
            for _i, kw in enumerate(kws[:9]):
                if dcols[_i % 3].button(f"⚡ {kw}", key=f"kw_{_i}", use_container_width=True):
                    q = f"Explain how '{kw}' is used in this paper."
                    st.session_state.messages.append({"role": "user", "content": q})
                    with st.spinner(f"Looking up '{kw}'…"):
                        try:
                            res = _engine.query(q, active_meta["vectorstore"], top_k)
                            st.session_state.messages.append({
                                "role": "assistant", "content": res["answer"],
                                "sources": res["sources"] if show_src else [],
                                "confidence": res["confidence"], "follow_ups": [],
                            })
                        except Exception as e:
                            st.session_state.messages.append({
                                "role": "assistant", "content": f"⚠️ {e}",
                                "sources": [], "confidence": "low", "follow_ups": [],
                            })
                    st.info("Answer added to Chat tab ↑")

# ── GAPS ───────────────────────────────────────────────────────────────────────
with t_gaps:
    col_id = st.session_state.active_col
    gap_key = f"gaps_{col_id}"
    if not api_key:
        st.info("🔑 Configure API key.")
    elif gap_key not in st.session_state:
        if st.button("🔍 Find Research Gaps", type="primary", key="gen_gaps"):
            with st.spinner("Analysing gaps…"):
                _analyzer.get_gaps(col_id, active_meta["vectorstore"])
                st.rerun()
    else:
        gaps = st.session_state[gap_key]
        for _key, _label, _cls in [
            ("explicit_gaps",    "📌 Explicit Gaps",      "tag"),
            ("implicit_gaps",    "🔎 Implicit Gaps",      "tag-amber"),
            ("future_directions","🚀 Future Directions",  "tag-green"),
            ("open_questions",   "❓ Open Questions",     "tag"),
        ]:
            items = gaps.get(_key, [])
            if items:
                st.markdown(f'<div class="sec-head">{_label}</div>', unsafe_allow_html=True)
                st.markdown("".join(f'<span class="tag {_cls}">{it}</span>' for it in items),
                            unsafe_allow_html=True)
                for _j, item in enumerate(items[:3]):
                    short = (item[:60] + "…") if len(item) > 60 else item
                    if st.button(f"💬 Discuss: {short}", key=f"gap_{_key}_{_j}",
                                 use_container_width=True):
                        q = f"What does the paper say about: '{item}'?"
                        st.session_state.messages.append({"role": "user", "content": q})
                        with st.spinner("Generating…"):
                            try:
                                res = _engine.query(q, active_meta["vectorstore"], top_k)
                                st.session_state.messages.append({
                                    "role": "assistant", "content": res["answer"],
                                    "sources": res["sources"] if show_src else [],
                                    "confidence": res["confidence"], "follow_ups": [],
                                })
                            except Exception as e:
                                st.session_state.messages.append({
                                    "role": "assistant", "content": f"⚠️ {e}",
                                    "sources": [], "confidence": "low", "follow_ups": [],
                                })
                        st.info("Response added to Chat tab ↑")
        if st.button("🔄 Regenerate", key="regen_gaps"):
            st.session_state.pop(gap_key, None)
            st.rerun()

# ── FLASHCARDS ─────────────────────────────────────────────────────────────────
with t_flash:
    col_id = st.session_state.active_col
    fc_key = f"fc_{col_id}"
    if not api_key:
        st.info("🔑 Configure API key.")
    elif fc_key not in st.session_state:
        if st.button("📚 Generate Flashcards", type="primary", key="gen_fc"):
            with st.spinner("Creating flashcards…"):
                _analyzer.get_flashcards(col_id, active_meta["vectorstore"])
                st.rerun()
    else:
        cards = st.session_state[fc_key]
        eli_key = f"eli15_{col_id}"
        head_col, btn_col = st.columns([3, 1])
        head_col.markdown(f'<div class="sec-head">{len(cards)} Flashcards</div>',
                          unsafe_allow_html=True)
        if btn_col.button("🧒 ELI15", key="eli15_btn", use_container_width=True):
            if eli_key not in st.session_state:
                with st.spinner("Simplifying…"):
                    _analyzer.get_eli15(col_id, active_meta["vectorstore"])
                    st.rerun()
        if eli_key in st.session_state:
            st.markdown(
                f'<div class="hero-card"><div class="hero-eyebrow">🧒 Explain Like I\'m 15</div>'
                f'<p style="font-size:14px;color:var(--text1);line-height:1.8;margin:0;">'
                f'{st.session_state[eli_key]}</p></div>',
                unsafe_allow_html=True,
            )
        if cards:
            lc, rc = st.columns(2)
            for _i, card in enumerate(cards):
                with (lc if _i % 2 == 0 else rc):
                    with st.expander(f"Card {_i+1} — {card.get('q','')[:55]}"):
                        st.markdown(
                            f'<div style="font-size:13px;color:var(--text0);line-height:1.7;">'
                            f'{card.get("a","")}</div>',
                            unsafe_allow_html=True,
                        )
        if st.button("🔄 Regenerate", key="regen_fc"):
            for _k in [fc_key, f"eli15_{col_id}"]:
                st.session_state.pop(_k, None)
            st.rerun()

# ── COMPARE ─────────────────────────────────────────────────────────────────────
with t_cmp:
    papers = list(st.session_state.collections.items())
    if len(papers) < 2:
        st.markdown(
            f'<div style="text-align:center;padding:50px 20px;">'
            f'<div style="font-size:40px;margin-bottom:12px;">⚖️</div>'
            f'<p style="font-size:15px;color:var(--text0);font-weight:600;">Upload 2+ papers to compare</p>'
            f'<p style="font-size:13px;color:var(--text1);">You have {len(papers)} paper indexed.</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
    elif not api_key:
        st.info("🔑 Configure API key.")
    else:
        pmap = {cid: meta["filename"] for cid, meta in papers}
        ca, cb = st.columns(2)
        sel_a = ca.selectbox("Paper A", list(pmap.keys()), format_func=lambda x: pmap[x],
                             key="cmp_a")
        sel_b = cb.selectbox("Paper B", [k for k in pmap if k != sel_a],
                             format_func=lambda x: pmap[x], key="cmp_b")
        if st.button("⚖️ Compare", type="primary", use_container_width=True, key="run_cmp"):
            with st.spinner("Comparing…"):
                _analyzer.get_compare(sel_a, sel_b, st.session_state.collections)
                st.rerun()
        cmp_store = st.session_state.get("compare_cache", {})
        cdata = cmp_store.get(f"{sel_a}__{sel_b}") or cmp_store.get(f"{sel_b}__{sel_a}")
        if cdata:
            l, r = st.columns(2)
            l.markdown(
                f'<div class="cmp-card"><div class="cmp-head">📄 {pmap[sel_a]}</div>'
                f'<div class="cmp-lbl">Contribution</div><div class="cmp-val">{cdata.get("a_contribution","")}</div>'
                f'<div class="cmp-lbl">Strength</div><div class="cmp-val">{cdata.get("a_strength","")}</div></div>',
                unsafe_allow_html=True,
            )
            r.markdown(
                f'<div class="cmp-card"><div class="cmp-head">📄 {pmap[sel_b]}</div>'
                f'<div class="cmp-lbl">Contribution</div><div class="cmp-val">{cdata.get("b_contribution","")}</div>'
                f'<div class="cmp-lbl">Strength</div><div class="cmp-val">{cdata.get("b_strength","")}</div></div>',
                unsafe_allow_html=True,
            )
            sl, sr = st.columns(2)
            with sl:
                st.markdown("**🤝 Similarities**")
                for s in cdata.get("similarities", []):
                    st.markdown(f"- {s}")
            with sr:
                st.markdown("**🔀 Differences**")
                for d in cdata.get("differences", []):
                    st.markdown(f"- {d}")
            st.markdown(
                f'<div class="info-card" style="margin-top:16px;">'
                f'<div class="info-eyebrow">💡 Synthesis & Reading Order</div>'
                f'<div class="info-text">{cdata.get("complementary","")}</div>'
                f'<div style="margin-top:8px;font-size:12px;color:var(--accent);font-family:var(--mono);">📖 {cdata.get("read_order","")}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

# ── EXPORT ─────────────────────────────────────────────────────────────────────
with t_exp:
    col_id = st.session_state.active_col
    fname = active_meta["filename"]
    msgs = st.session_state.messages
    e1, e2, e3 = st.columns(3)

    with e1:
        st.markdown(
            '<div class="info-card"><div class="info-eyebrow">💬 Chat</div>'
            f'<div class="info-text">{len([m for m in msgs if m["role"]=="user"])} questions</div></div>',
            unsafe_allow_html=True,
        )
        if msgs:
            st.download_button(
                "⬇️ Chat (.md)", data=Exporter.chat_to_markdown(msgs, fname),
                file_name=f"chat_{col_id}.md", mime="text/markdown",
                key="dl_chat", use_container_width=True,
            )
        else:
            st.caption("No messages yet.")

    with e2:
        s = st.session_state.summaries.get(col_id)
        st.markdown(
            '<div class="info-card"><div class="info-eyebrow">📋 Summary</div>'
            '<div class="info-text">Structured analysis</div></div>',
            unsafe_allow_html=True,
        )
        if s:
            st.download_button(
                "⬇️ Summary (.md)", data=Exporter.summary_to_markdown(s, fname),
                file_name=f"summary_{col_id}.md", mime="text/markdown",
                key="dl_sum_e", use_container_width=True,
            )
        else:
            st.caption("Generate summary first.")

    with e3:
        st.markdown(
            '<div class="info-card"><div class="info-eyebrow">🗂 Full Session (JSON)</div>'
            '<div class="info-text">Messages · Summary · Gaps · Flashcards</div></div>',
            unsafe_allow_html=True,
        )
        session_data = {
            "exported_at": datetime.datetime.now().isoformat(),
            "paper": fname,
            "chunks": active_meta["chunks"],
            "messages": [
                {"role": m["role"], "content": m["content"],
                 "confidence": m.get("confidence"), "sources": m.get("sources", [])}
                for m in msgs
            ],
            "summary": st.session_state.summaries.get(col_id, {}),
            "gaps": st.session_state.get(f"gaps_{col_id}", {}),
            "flashcards": st.session_state.get(f"fc_{col_id}", []),
        }
        st.download_button(
            "⬇️ Session (.json)", data=json.dumps(session_data, indent=2),
            file_name=f"session_{col_id}.json", mime="application/json",
            key="dl_json", use_container_width=True,
        )

    st.markdown('<div class="sec-head">Session Stats</div>', unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Questions", len([m for m in msgs if m["role"] == "user"]))
    s2.metric("Summary",   "✓" if col_id in st.session_state.summaries else "–")
    s3.metric("Gaps",      "✓" if f"gaps_{col_id}" in st.session_state else "–")
    s4.metric("Flashcards", len(st.session_state.get(f"fc_{col_id}", [])))
