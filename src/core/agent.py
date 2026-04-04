"""LangChain tool-calling agent: document search + Tavily web search."""

from __future__ import annotations

from typing import Any

from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import StructuredTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults

from core.config import (
    TOOL_NAME_DOCUMENT_SEARCH,
    TOOL_NAME_WEB_SEARCH,
    Settings,
)

# Stable defaults (tune in code if needed; not exposed via .env)
RAG_CHUNK_DISPLAY_SEPARATOR = "\n\n---\n\n"
DOCUMENT_SEARCH_EMPTY_MESSAGE = (
    "No matching passages were found in the uploaded documents."
)
WEB_SEARCH_EXCERPT_SEPARATOR = "\n\n---\n\n"
_GEMINI_CHAT_TEMPERATURE = 0.2
_GEMINI_CHAT_STREAMING = True
_TAVILY_MAX_RESULTS = 4
_TAVILY_SEARCH_DEPTH = "advanced"


def stringify_agent_output(output: Any) -> str:
    """
    Turn AgentExecutor ``output`` into plain text for the UI.

    Gemini (and some LangChain integrations) return ``AIMessage.content`` as a
    list of blocks such as ``{'type': 'text', 'text': '...', 'extras': {...}}``
    (thought signatures, etc.). The tools agent forwards that list unchanged, so
    we extract only human-visible text and concatenate string fragments.
    """
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    if isinstance(output, AIMessage):
        return stringify_agent_output(output.content)
    if isinstance(output, list):
        parts: list[str] = []
        for block in output:
            parts.append(_stringify_content_block(block))
        return "".join(parts)
    return str(output)


def _stringify_content_block(block: Any) -> str:
    if isinstance(block, str):
        return block
    if isinstance(block, dict):
        if block.get("type") == "text" and "text" in block:
            return str(block["text"])
        if "text" in block:
            return str(block["text"])
        return ""
    return str(block)


def _tool_name(action: Any) -> str:
    return getattr(action, "tool", "") or ""


def _extract_tavily_result_rows(observation: Any) -> list[dict] | None:
    """
    TavilySearchResults returns ``(clean_results, raw_artifact)`` where
    ``clean_results`` is a list of dicts with title, url, content, score.
    """
    if isinstance(observation, tuple) and len(observation) >= 1:
        first = observation[0]
        if isinstance(first, list) and all(isinstance(x, dict) for x in first):
            return first
    if isinstance(observation, list) and all(
        isinstance(x, dict) for x in observation
    ):
        return observation
    return None


def _format_tavily_results_for_ui(rows: list[dict], block_sep: str) -> str:
    """Human-readable markdown blocks for Streamlit (one search call)."""
    blocks: list[str] = []
    for r in rows:
        title = str(r.get("title") or "Untitled").strip()
        url = str(r.get("url") or "").strip()
        content = str(r.get("content") or "").strip()
        if url:
            head = f"### [{title}]({url})"
        else:
            head = f"### {title}"
        body = content if content else "*No snippet returned.*"
        blocks.append(f"{head}\n\n{body}")
    return block_sep.join(blocks)


def _web_tool_observation_to_text(observation: Any, block_sep: str) -> str:
    rows = _extract_tavily_result_rows(observation)
    if rows is not None:
        return _format_tavily_results_for_ui(rows, block_sep)
    if isinstance(observation, str):
        return observation.strip()
    return str(observation).strip()


def parse_agent_sources(intermediate_steps: list | None) -> dict[str, Any]:
    """
    Derive UI metadata from AgentExecutor intermediate steps.

    Returns keys: label (str), document_chunks (list[str]), web_excerpt (str).
    Label is always a single category: documents take precedence over web when
    both were used in the same turn (expanders still show passages and web snippets).
    """
    used_docs = False
    used_web = False
    document_chunks: list[str] = []
    web_parts: list[str] = []
    doc_tool = TOOL_NAME_DOCUMENT_SEARCH
    web_tool = TOOL_NAME_WEB_SEARCH
    chunk_sep = RAG_CHUNK_DISPLAY_SEPARATOR
    empty_substr = DOCUMENT_SEARCH_EMPTY_MESSAGE

    if not intermediate_steps:
        return {
            "label": "No tools used",
            "document_chunks": [],
            "web_excerpt": "",
        }

    wsep = WEB_SEARCH_EXCERPT_SEPARATOR

    for action, observation in intermediate_steps:
        name = _tool_name(action)

        if name == doc_tool:
            obs = (
                observation
                if isinstance(observation, str)
                else str(observation)
            )
            # Only count as "documents" if retrieval returned real passages — the
            # agent often invokes document search first even for web-only questions,
            # which yields DOCUMENT_SEARCH_EMPTY_MESSAGE alone.
            if obs and empty_substr not in obs:
                for part in obs.split(chunk_sep):
                    p = part.strip()
                    if p:
                        document_chunks.append(p)
                        used_docs = True
        elif name == web_tool:
            used_web = True
            excerpt = _web_tool_observation_to_text(observation, wsep)
            if excerpt:
                web_parts.append(excerpt)

    if used_docs:
        label = "Your documents"
    elif used_web:
        label = "Web search"
    else:
        label = "No retrieval"

    if label == "Web search":
        document_chunks = []

    return {
        "label": label,
        "document_chunks": document_chunks,
        "web_excerpt": wsep.join(web_parts) if web_parts else "",
    }


def _format_doc_block(text: str, meta: dict) -> str:
    label = meta.get("citation_label") or meta.get("source", "document")
    page = meta.get("page")
    src = meta.get("source", "")
    if page is not None:
        header = f"[Document: {src} | Page: {int(page) + 1} | Ref: {label}]"
    else:
        header = f"[Document: {src} | Ref: {label}]"
    return f"{header}\n{text.strip()}"


def build_document_search_tool(retriever: BaseRetriever, settings: Settings):
    def _search_uploaded_documents(query: str) -> str:
        docs = retriever.invoke(query)
        if not docs:
            return DOCUMENT_SEARCH_EMPTY_MESSAGE
        parts = [_format_doc_block(d.page_content, d.metadata) for d in docs]
        return RAG_CHUNK_DISPLAY_SEPARATOR.join(parts)

    return StructuredTool.from_function(
        func=_search_uploaded_documents,
        name=TOOL_NAME_DOCUMENT_SEARCH,
        description=settings.document_search_tool_description,
    )


def build_agent_executor(
    *,
    settings: Settings,
    retriever: BaseRetriever,
) -> AgentExecutor:
    llm = ChatGoogleGenerativeAI(
        model=settings.gemini_chat_model,
        google_api_key=settings.google_api_key,
        temperature=_GEMINI_CHAT_TEMPERATURE,
        max_retries=settings.gemini_max_retries,
        streaming=_GEMINI_CHAT_STREAMING,
    )

    doc_tool = build_document_search_tool(retriever, settings)
    web_tool = TavilySearchResults(
        max_results=_TAVILY_MAX_RESULTS,
        search_depth=_TAVILY_SEARCH_DEPTH,
        tavily_api_key=settings.tavily_api_key,
        name=TOOL_NAME_WEB_SEARCH,
        description=settings.tavily_tool_description,
    )
    tools = [doc_tool, web_tool]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", settings.agent_system_prompt),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=settings.agent_max_iterations,
        return_intermediate_steps=True,
    )


def session_messages_to_lc(messages: list[dict]) -> list[BaseMessage]:
    """Convert Streamlit-style messages to LangChain messages (excludes latest user message if caller passes it separately)."""
    out: list[BaseMessage] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if role == "user":
            out.append(HumanMessage(content=content))
        elif role == "assistant":
            out.append(AIMessage(content=content))
    return out
