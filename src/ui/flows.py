"""Streamlit flows: index PDFs (step 1), chat with agent (step 2)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import streamlit as st
from langchain_community.callbacks.streamlit import LLMThoughtLabeler, StreamlitCallbackHandler

from core.agent import (
    build_agent_executor,
    parse_agent_sources,
    session_messages_to_lc,
    stringify_agent_output,
)
from core.config import Settings
from core.ingestion import load_and_chunk_pdfs
from core.metrics import compute_ingestion_metrics
from core.retrieval import build_retriever
from core.vectorstore import add_documents_to_session, get_vectorstore

from ui.layout import (
    render_assistant_answer_block,
    render_step_banner,
    scroll_chat_into_view,
)


def _retriever_cache_key(
    settings: Settings,
    selected: tuple[str, ...],
    indexed_sources: tuple[str, ...],
) -> tuple:
    """Cache key: selection + corpus + retrieval-related settings (filter derives from selection)."""
    return (
        selected,
        indexed_sources,
        settings.retrieval_k,
        settings.use_multi_query,
        settings.use_rerank,
        settings.retrieval_fetch_multiplier,
        settings.gemini_max_retries,
    )


def sync_agent_retriever(settings: Settings) -> None:
    """Rebuild agent when retrieval filter or corpus changes (metadata filter + rerank config)."""
    if not st.session_state.get("indexed"):
        return
    indexed_sources = tuple(sorted(st.session_state.get("indexed_sources") or []))
    raw_sel = st.session_state.get("retrieval_filter_multiselect") or []
    selected = tuple(sorted(s for s in raw_sel if s in set(indexed_sources)))
    meta_filter: dict | None = None
    if selected:
        meta_filter = {"source": {"$in": list(selected)}}
    key = _retriever_cache_key(settings, selected, indexed_sources)
    if (
        st.session_state.get("agent_retriever_key") == key
        and st.session_state.get("agent_executor") is not None
    ):
        return
    vs = get_vectorstore(
        settings=settings,
        session_id=st.session_state.session_id,
    )
    retriever = build_retriever(
        vectorstore=vs,
        settings=settings,
        metadata_filter=meta_filter,
    )
    st.session_state.agent_executor = build_agent_executor(
        settings=settings,
        retriever=retriever,
    )
    st.session_state.agent_retriever_key = key


_GEMINI_429_HELP = """
**Gemini quota exceeded (429).** Google refused the request because of **rate or usage limits** on **chat** (`generate_content`).

On the **free tier**, daily limits per model are small. This **agent** often needs **several** chat calls **per message** (tool rounds plus the final answer), so the cap is easy to hit.

**What you can do**

1. **Enable billing** on the Google AI / Gemini project tied to your key ([Google AI Studio](https://aistudio.google.com/) → API key / billing).
2. **Try another chat model** in `.env` — `GEMINI_CHAT_MODEL=…` ([model list](https://ai.google.dev/gemini-api/docs/models)); quotas are **per model**.
3. **Lower `AGENT_MAX_ITERATIONS`** (e.g. `3`) — fewer tool loops per reply.
4. **Set `USE_MULTI_QUERY=false`** — multi-query adds extra chat calls each retrieval.
5. **Set `GEMINI_MAX_RETRIES=0`** — no retries on failure (default is `1`).
6. If the error mentions **PerDay**, wait for the daily reset.

Embeddings use a **separate** quota; **this 429 is chat only.** [Gemini rate limits](https://ai.google.dev/gemini-api/docs/rate-limits)
"""

# Stored in chat history: self-contained on rerun (no ephemeral UI).
_GEMINI_429_CHAT_SUMMARY = (
    "**Gemini chat quota (429)** — `generate_content` hit a rate or daily limit; this reply did not finish.\n\n"
    "**Try:** billing or a fresh key in [Google AI Studio](https://aistudio.google.com/); "
    "another `GEMINI_CHAT_MODEL` ([models](https://ai.google.dev/gemini-api/docs/models)); "
    "lower `AGENT_MAX_ITERATIONS`; `USE_MULTI_QUERY=false`; `GEMINI_MAX_RETRIES=0`. "
    "See README **Troubleshooting** and [Gemini rate limits](https://ai.google.dev/gemini-api/docs/rate-limits)."
)


class _AssistantThoughtLabeler(LLMThoughtLabeler):
    @staticmethod
    def get_initial_label() -> str:
        return "**Working…**"


def _is_likely_gemini_429(error_text: str) -> bool:
    if "429" not in error_text:
        return False
    low = error_text.lower()
    return (
        "resource_exhausted" in low
        or "quota" in low
        or "exhausted" in low
        or "rate limit" in low
        or "too many requests" in low
    )


def _explain_gemini_429() -> None:
    st.error("Gemini quota or rate limit (429)")
    st.markdown(_GEMINI_429_HELP)


def render_indexing_flow(settings: Settings) -> None:
    with st.container(border=True):
        render_step_banner(1, "Add your PDFs")
        st.caption(
            f"Up to **{settings.max_upload_files}** PDFs, **{settings.max_pages_per_file}** pages per file "
            f"(see <code>.env</code> to change). Files are split into overlapping chunks, embedded, and stored for search."
        )
        uploaded = st.file_uploader(
            "Drop PDFs here",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            help=f"Max {settings.max_upload_files} files, {settings.max_pages_per_file} pages per file.",
        )
        col_idx, col_status = st.columns([1, 1], gap="medium")
        with col_idx:
            index_clicked = st.button(
                "Index PDFs & enable chat",
                type="primary",
                disabled=not uploaded,
                use_container_width=True,
            )
        with col_status:
            if st.session_state.indexed:
                st.success(f"**{st.session_state.chunk_total}** passages indexed—ready to chat.")
            else:
                st.info("Index your PDFs to unlock the assistant below.", icon="✨")

        if index_clicked and uploaded:
            if len(uploaded) > settings.max_upload_files:
                st.error(f"Too many files (max {settings.max_upload_files}).")
            else:
                paths: list[Path] = []
                original_names: list[str] = []
                try:
                    progress = st.progress(0, text="Preparing files…")
                    for f in uploaded:
                        original_names.append(f.name)
                        suffix = Path(f.name).suffix or ".pdf"
                        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
                        os.close(fd)
                        p = Path(tmp_path)
                        p.write_bytes(f.getvalue())
                        paths.append(p)
                    progress.progress(0.2, text="Loading PDFs and chunking…")
                    docs = load_and_chunk_pdfs(
                        uploaded_paths=paths,
                        source_filenames=original_names,
                        session_id=st.session_state.session_id,
                        settings=settings,
                    )
                    progress.progress(0.45, text="Connecting to vector store…")
                    vs = get_vectorstore(
                        settings=settings,
                        session_id=st.session_state.session_id,
                    )
                    progress.progress(0.65, text="Embedding & indexing passages…")
                    n = add_documents_to_session(vs, docs)
                    st.session_state.chunk_total = n
                    st.session_state.indexed = True
                    st.session_state.indexed_sources = sorted(
                        {Path(name).name for name in original_names}
                    )
                    st.session_state.ingestion_metrics = compute_ingestion_metrics(docs)
                    if "retrieval_filter_multiselect" in st.session_state:
                        del st.session_state.retrieval_filter_multiselect
                    st.session_state.pop("agent_retriever_key", None)
                    progress.progress(0.88, text="Wiring retrieval & tools…")
                    sync_agent_retriever(settings)
                    progress.progress(1.0, text="Done")
                    st.success(f"**{n}** passages indexed. You can start asking questions below.")
                    progress.empty()
                except ValueError as err:
                    st.error(str(err))
                finally:
                    for p in paths:
                        try:
                            p.unlink(missing_ok=True)
                        except OSError:
                            pass


def render_chat_flow(settings: Settings) -> None:
    st.divider()
    render_step_banner(2, "Chat with your corpus")
    st.caption(
        "The model may call retrieval, web search, or both. Replies can include a source badge and expandable "
        "passages from your PDFs."
    )

    if st.session_state.indexed and not st.session_state.messages:
        st.info(
            "**Ask something**\n\n"
            "Try a question about your PDFs, or something that needs the web—the "
            "**Source** line above each answer shows what was used.",
            icon="💬",
        )

    _chat_ph = (
        "Index PDFs in Step 1 to start chatting…"
        if not st.session_state.indexed
        else "Ask about your PDFs, or anything you’d verify on the web…"
    )
    if prompt := st.chat_input(_chat_ph, disabled=not st.session_state.indexed):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.pending_reply = True
        st.rerun()

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            if m["role"] == "assistant":
                render_assistant_answer_block(
                    m.get("content", ""),
                    m.get("sources"),
                )
            else:
                st.markdown(m["content"])

    if st.session_state.get("pending_reply") and st.session_state.messages:
        scroll_chat_into_view()

    if st.session_state.get("pending_reply"):
        agent = st.session_state.get("agent_executor")
        if not agent:
            st.session_state.pending_reply = False
            st.error("Agent not ready. Index documents first.")
            st.stop()

        st.session_state.pending_reply = False
        user_text = st.session_state.messages[-1]["content"]
        history = session_messages_to_lc(st.session_state.messages[:-1])

        text = ""
        sources: dict | None = None
        with st.chat_message("assistant"):
            try:
                with st.status("Working on your answer…", expanded=False) as status:
                    status.update(label="Calling model and tools…", state="running")
                    cb = StreamlitCallbackHandler(
                        st.container(),
                        thought_labeler=_AssistantThoughtLabeler(),
                    )
                    res = agent.invoke(
                        {"input": user_text, "chat_history": history},
                        config={"callbacks": [cb]},
                    )
                    status.update(label="Done", state="complete")
                text = stringify_agent_output(res.get("output", ""))
                sources = parse_agent_sources(res.get("intermediate_steps"))
                render_assistant_answer_block(text, sources)
            except Exception as e:
                err = f"{e!s}"
                if _is_likely_gemini_429(err):
                    _explain_gemini_429()
                    with st.expander("Technical details"):
                        st.code(err)
                    text = _GEMINI_429_CHAT_SUMMARY
                else:
                    text = f"Error: {err}"
                    st.error(text)

        assistant_msg: dict = {"role": "assistant", "content": text}
        if sources is not None:
            assistant_msg["sources"] = sources
        st.session_state.messages.append(assistant_msg)
        scroll_chat_into_view()
        st.rerun()
