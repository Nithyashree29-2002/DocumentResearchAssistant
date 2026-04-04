"""Global CSS, intro, sidebar, step banners, ingestion stats, assistant message UI."""

from __future__ import annotations

import html
import re
import uuid

import streamlit as st
import streamlit.components.v1 as st_components

_APP_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,400;0,500;0,600;0,700;1,400&display=swap');
    .block-container {
        padding-top: 2.75rem !important;
        padding-bottom: 3rem !important;
        max-width: 56rem !important;
    }
    @media (min-width: 1200px) {
        .block-container { max-width: 72rem !important; }
    }
    [data-testid="stMain"] {
        overflow-x: visible !important;
    }
    [data-testid="stAppViewContainer"] h1 {
        font-family: 'Plus Jakarta Sans', system-ui, sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.04em !important;
        line-height: 1.12 !important;
        margin-bottom: 0.2rem !important;
    }
    [data-testid="stAppViewContainer"] h3,
    [data-testid="stAppViewContainer"] h4,
    [data-testid="stAppViewContainer"] h5 {
        font-family: 'Plus Jakarta Sans', system-ui, sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        overflow: visible !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        border-radius: 16px !important;
        overflow: visible !important;
        padding-top: 0.6rem !important;
    }
    .ra-card-top-accent {
        height: 5px;
        width: 100%;
        max-width: 8rem;
        border-radius: 999px;
        margin: 0.15rem 0 0.9rem 0;
        background: linear-gradient(90deg, #0f766e, #2dd4bf, #38bdf8);
        opacity: 0.95;
    }
    @media (prefers-reduced-motion: no-preference) {
        .ra-feature-card:hover {
            transform: translateY(-2px);
        }
    }
    [data-testid="stFileUploaderDropzone"] {
        border-radius: 14px !important;
        border-width: 2px !important;
        border-style: dashed !important;
        border-color: rgba(15, 118, 110, 0.35) !important;
        background: rgba(15, 118, 110, 0.05) !important;
        transition: border-color 0.2s ease, background 0.2s ease !important;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: rgba(15, 118, 110, 0.6) !important;
        background: rgba(15, 118, 110, 0.09) !important;
    }
    [data-testid="baseButton-primary"] {
        border-radius: 10px !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 8px rgba(15, 118, 110, 0.25) !important;
    }
    .ra-features-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.75rem;
        margin: 0.35rem 0 0.15rem 0;
    }
    @media (max-width: 768px) {
        .ra-features-grid { grid-template-columns: 1fr; }
    }
    .ra-feature-card {
        border-radius: 14px;
        padding: 1rem 1.1rem;
        border: 1px solid rgba(128, 128, 128, 0.2);
        background: var(--secondary-background-color);
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
        transition: box-shadow 0.2s ease, border-color 0.2s ease, transform 0.2s ease;
    }
    .ra-feature-card:hover {
        border-color: rgba(15, 118, 110, 0.4);
        box-shadow: 0 8px 24px rgba(15, 118, 110, 0.12);
    }
    .ra-feature-title {
        font-family: 'Plus Jakarta Sans', system-ui, sans-serif;
        font-weight: 700;
        font-size: 0.9rem;
        margin: 0 0 0.35rem 0;
        color: var(--text-color);
        letter-spacing: -0.02em;
    }
    .ra-feature-desc {
        font-size: 0.8rem;
        line-height: 1.5;
        margin: 0;
        color: var(--text-color);
        opacity: 0.88;
    }
    span.ra-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.045em;
        padding: 0.4rem 0.7rem;
        border-radius: 999px;
        margin-bottom: 0.75rem;
        border: 1px solid rgba(0, 0, 0, 0.06);
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    }
    span.ra-pill--docs { background: rgba(16, 124, 65, 0.12); color: #0d5c36; }
    span.ra-pill--web { background: rgba(200, 120, 0, 0.14); color: #8a5a00; }
    span.ra-pill--muted { background: rgba(120, 120, 120, 0.12); color: #555; }
    div.ra-chunk-hdr {
        font-family: 'Plus Jakarta Sans', system-ui, sans-serif;
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: #5a6570;
        margin-bottom: 0.4rem;
    }
    div.ra-chunk-cite {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, monospace;
        font-size: 0.76rem;
        color: #3d5a80;
        background: rgba(61, 90, 128, 0.08);
        padding: 0.5rem 0.65rem;
        border-radius: 8px;
        margin-bottom: 0.65rem;
        word-break: break-word;
        border: 1px solid rgba(61, 90, 128, 0.12);
    }
    section[data-testid="stSidebar"] .block-container { padding-top: 1rem !important; }
    section[data-testid="stSidebar"] h3 {
        font-family: 'Plus Jakarta Sans', system-ui, sans-serif !important;
        font-weight: 700 !important;
    }
</style>
"""

_CHUNK_CITE_RE = re.compile(
    r"^\s*(\[Document:[^\]]+\])\s*\n?(.*)$",
    re.DOTALL,
)


def inject_app_styles() -> None:
    st.markdown(_APP_CSS, unsafe_allow_html=True)


def init_session_state() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "indexed" not in st.session_state:
        st.session_state.indexed = False
    if "chunk_total" not in st.session_state:
        st.session_state.chunk_total = 0
    if "ingestion_metrics" not in st.session_state:
        st.session_state.ingestion_metrics = None
    if "indexed_sources" not in st.session_state:
        st.session_state.indexed_sources = []


def render_app_header() -> None:
    with st.container(border=True):
        st.markdown(
            '<div class="ra-card-top-accent" aria-hidden="true"></div>',
            unsafe_allow_html=True,
        )
        st.title("Research Assistant")
        st.caption("RAG · your PDFs + optional web search")
        st.divider()
        st.markdown(
            "Upload PDFs and ask questions in plain language. The app searches your files first, "
            "uses the web when that helps, and can show **sources** and **retrieved passages**."
        )
        st.markdown(
            "**Stack:** `LangChain` agent with retrieval + web tools, embeddings and a vector store, "
            "conversation memory per browser tab, `Streamlit` UI."
        )
        st.divider()
        st.markdown(
            '<div class="ra-features-grid">'
            '<div class="ra-feature-card">'
            '<div class="ra-feature-title">📄 Your PDFs</div>'
            '<p class="ra-feature-desc">Chunked, embedded text with file and page citations when available.</p>'
            "</div>"
            '<div class="ra-feature-card">'
            '<div class="ra-feature-title">🌐 Web</div>'
            '<p class="ra-feature-desc">Search when documents do not cover the question.</p>'
            "</div>"
            '<div class="ra-feature-card">'
            '<div class="ra-feature-title">💬 Session chat</div>'
            '<p class="ra-feature-desc">Follow-ups stay in context until you reset the session.</p>'
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.caption("Use the **← sidebar** to reset this tab (clears chat and indexed docs).")
    st.markdown("")


def render_step_banner(step_num: int, label: str) -> None:
    safe_label = html.escape(label)
    st.markdown(
        f'<div style="'
        "font-family:system-ui,-apple-system,'Plus Jakarta Sans',sans-serif;"
        "display:flex;align-items:center;gap:0.75rem;"
        "background:linear-gradient(145deg,#115e59 0%,#0f766e 42%,#14b8a6 100%);"
        "color:#f8fafc;"
        "padding:0.85rem 1.15rem;"
        "border-radius:16px;"
        "margin:0 0 1rem 0;"
        "border:1px solid rgba(0,0,0,0.18);"
        "box-shadow:0 1px 0 rgba(255,255,255,0.14) inset,0 8px 28px rgba(6,95,70,0.3);"
        '">'
        f'<span style="font-size:0.68rem;font-weight:800;letter-spacing:0.14em;opacity:0.85;">'
        f"STEP {step_num}</span>"
        f'<span style="font-size:1.06rem;font-weight:700;letter-spacing:-0.03em;line-height:1.3;">{safe_label}</span>'
        "</div>",
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### Session")
        st.caption("Clears chat history and the indexed documents for **this tab** only.")
        if st.button("Reset session", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

        if st.session_state.get("indexed") and st.session_state.get("indexed_sources"):
            st.divider()
            st.markdown("### Retrieval scope")
            st.caption(
                "Restrict document search to selected PDFs (Pinecone metadata `source`). "
                "Leave unselected to search the full indexed corpus."
            )
            st.multiselect(
                "Included documents",
                options=st.session_state.indexed_sources,
                key="retrieval_filter_multiselect",
            )


def render_ingestion_metrics_section() -> None:
    if not st.session_state.indexed:
        return
    with st.expander("Chunk & ingestion stats", expanded=False):
        st.caption("Numbers from the last successful index (chunk counts, lengths, metadata coverage).")
        st.markdown("##### Last successful index")
        im = st.session_state.get("ingestion_metrics")
        if im:
            u1, u2, u3, u4 = st.columns(4)
            u1.metric("Chunks", im["n_chunks"])
            u2.metric("Unique PDFs", im["n_unique_sources"])
            u3.metric("Avg chars / chunk", im["chunk_length_mean"])
            u4.metric("Chunks w/ page meta", f'{im["pct_chunks_with_page"]}%')
            with st.expander("Full ingestion stats", expanded=False):
                st.json(im)
        else:
            st.caption("Re-index documents to record ingestion metrics for this session.")


def _split_chunk_citation(raw: str) -> tuple[str | None, str]:
    m = _CHUNK_CITE_RE.match(raw.strip())
    if not m:
        return None, raw.strip()
    cite, rest = m.group(1), (m.group(2) or "").strip()
    return cite, rest if rest else ""


def _render_source_pill(label: str) -> None:
    if label == "Your documents":
        css, icon = "ra-pill--docs", "📄"
    elif label == "Web search":
        css, icon = "ra-pill--web", "🌐"
    elif label in ("No tools used", "No retrieval"):
        css, icon = "ra-pill--muted", "○"
    else:
        css, icon = "ra-pill--muted", "○"
    safe = html.escape(label)
    st.markdown(
        f'<span class="ra-pill {css}">{icon} Source · {safe}</span>',
        unsafe_allow_html=True,
    )


def _render_retrieved_chunks(chunks: list[str]) -> None:
    if not chunks:
        return
    n = len(chunks)
    with st.expander(f"Retrieved passages ({n})", expanded=False):
        st.caption("Grounding for this answer—open during the demo to show exactly what was retrieved.")
        for i, ch in enumerate(chunks, start=1):
            cite, body = _split_chunk_citation(ch)
            with st.container(border=True):
                st.markdown(
                    f'<div class="ra-chunk-hdr">Passage {i} of {n}</div>',
                    unsafe_allow_html=True,
                )
                if cite:
                    st.markdown(
                        f'<div class="ra-chunk-cite">{html.escape(cite)}</div>',
                        unsafe_allow_html=True,
                    )
                if body:
                    st.markdown(body)
                elif not cite:
                    st.markdown(ch)


def _render_web_excerpts(web_ex: str) -> None:
    web_ex = web_ex.strip()
    if not web_ex:
        return
    parts = [p.strip() for p in web_ex.split("\n\n---\n\n") if p.strip()]
    if not parts:
        return
    n = len(parts)
    with st.expander(f"Web sources ({n})", expanded=False):
        st.caption("Snippets from Tavily search used for this answer.")
        for i, part in enumerate(parts, start=1):
            with st.container(border=True):
                st.markdown(
                    f'<div class="ra-chunk-hdr">Result {i} of {n}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(part)


def render_assistant_answer_block(content: str, sources: dict | None) -> None:
    if sources and isinstance(sources, dict):
        _render_source_pill(sources.get("label", ""))
    st.markdown(content)
    if not sources or not isinstance(sources, dict):
        return
    label = sources.get("label", "")
    chunks = sources.get("document_chunks") or []
    # Web-only answers: never show PDF passages (avoids stray chunks vs pill mismatch).
    if chunks and label != "Web search":
        st.markdown("")
        _render_retrieved_chunks(chunks)
    web_ex = (sources.get("web_excerpt") or "").strip()
    if web_ex:
        st.markdown("")
        _render_web_excerpts(web_ex)


def scroll_chat_into_view() -> None:
    html_js = """
<script>
(function () {
  function raScrollMainToBottom() {
    try {
      var doc = window.parent.document;
      if (!doc || !doc.body) return;
      var selectors = [
        '[data-testid="stAppViewContainer"]',
        'section[data-testid="stMain"]',
        'section.main',
        '.stMain',
        '.main'
      ];
      for (var i = 0; i < selectors.length; i++) {
        var el = doc.querySelector(selectors[i]);
        if (el) {
          el.scrollTop = el.scrollHeight;
        }
      }
      doc.documentElement.scrollTop = doc.body.scrollHeight;
      window.parent.scrollTo(0, doc.body.scrollHeight);
    } catch (e) {}
  }
  raScrollMainToBottom();
  requestAnimationFrame(raScrollMainToBottom);
  setTimeout(raScrollMainToBottom, 80);
  setTimeout(raScrollMainToBottom, 250);
  setTimeout(raScrollMainToBottom, 600);
})();
</script>
"""
    st_components.html(html_js, height=0, width=0, scrolling=False)
