"""Streamlit app: adds ``src/`` to path, then page shell + layout + flows."""

from __future__ import annotations

import html
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import streamlit as st

from core.config import get_settings
from ui.flows import render_chat_flow, render_indexing_flow, sync_agent_retriever
from ui.layout import (
    init_session_state,
    inject_app_styles,
    render_app_header,
    render_ingestion_metrics_section,
    render_sidebar,
)


def main() -> None:
    st.set_page_config(
        page_title="Research Assistant",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_app_styles()
    init_session_state()
    render_app_header()

    try:
        settings = get_settings()
    except ValueError as e:
        st.error(
            "**Configuration needed**\n\n"
            f"{html.escape(str(e))}\n\n"
            "Copy `.env.example` to `.env` and add your API keys."
        )
        return

    render_sidebar()
    sync_agent_retriever(settings)
    render_indexing_flow(settings)
    sync_agent_retriever(settings)
    render_ingestion_metrics_section()
    render_chat_flow(settings)


if __name__ == "__main__":
    main()
