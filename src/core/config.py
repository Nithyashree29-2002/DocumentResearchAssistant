"""Environment-backed settings (see `.env.example`). Defaults live here only."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Fixed tool names (referenced in prompts and agent); change here + prompt files together.
TOOL_NAME_DOCUMENT_SEARCH = "search_uploaded_documents"
TOOL_NAME_WEB_SEARCH = "internet_search"


def _normalize_gemini_chat_model(name: str) -> str:
    m = name.strip()
    if m.startswith("models/"):
        m = m[7:]
    # "gemini-2.5" alone is not a valid generateContent model id
    if m == "gemini-2.5":
        return "gemini-2.5-flash"
    return m


def _truthy(val: str | None, default: bool = False) -> bool:
    if val is None or val.strip() == "":
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _resolve_path(path_str: str) -> Path:
    p = Path(path_str.strip())
    if p.is_absolute():
        return p
    return _PROJECT_ROOT / p


def _env_override_or_file(
    env_inline: str,
    file_path_env: str,
    default_file_rel: str,
) -> str:
    inline = os.getenv(env_inline, "").strip()
    if inline:
        return inline
    path_key = os.getenv(file_path_env, "").strip() or default_file_rel
    return _read_text_file(_resolve_path(path_key))


@dataclass(frozen=True)
class Settings:
    google_api_key: str
    pinecone_api_key: str
    pinecone_index_name: str
    tavily_api_key: str
    gemini_chat_model: str
    gemini_embedding_model: str
    use_multi_query: bool
    max_upload_files: int
    max_pages_per_file: int
    retrieval_k: int
    use_rerank: bool
    retrieval_fetch_multiplier: int
    pinecone_auto_create_index: bool
    pinecone_cloud: str
    pinecone_region: str
    pinecone_embedding_dimension: int
    pinecone_metric: str
    agent_max_iterations: int
    gemini_max_retries: int
    chunk_size: int
    chunk_overlap: int
    agent_system_prompt: str
    tavily_tool_description: str
    document_search_tool_description: str


@lru_cache
def get_settings() -> Settings:
    google = (
        os.getenv("GOOGLE_API_KEY", "").strip()
        or os.getenv("GEMINI_API_KEY", "").strip()
    )
    pc = os.getenv("PINECONE_API_KEY", "").strip()
    idx = os.getenv("PINECONE_INDEX_NAME", "rag-research-assistant").strip()
    tv = os.getenv("TAVILY_API_KEY", "").strip()
    if not google:
        raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY is required")
    if not pc:
        raise ValueError("PINECONE_API_KEY is required")
    if not tv:
        raise ValueError("TAVILY_API_KEY is required")

    system_template = _env_override_or_file(
        "AGENT_SYSTEM_PROMPT",
        "AGENT_SYSTEM_PROMPT_FILE",
        "prompts/agent_system.txt",
    )
    agent_system_prompt = system_template.format(
        DOCUMENT_SEARCH_TOOL_NAME=TOOL_NAME_DOCUMENT_SEARCH,
        WEB_SEARCH_TOOL_NAME=TOOL_NAME_WEB_SEARCH,
    )

    tavily_desc = _env_override_or_file(
        "TAVILY_TOOL_DESCRIPTION",
        "TAVILY_TOOL_DESCRIPTION_FILE",
        "prompts/tavily_tool_description.txt",
    )
    doc_tool_desc = _env_override_or_file(
        "DOCUMENT_SEARCH_TOOL_DESCRIPTION",
        "DOCUMENT_SEARCH_TOOL_DESCRIPTION_FILE",
        "prompts/document_search_tool_description.txt",
    )

    return Settings(
        google_api_key=google,
        pinecone_api_key=pc,
        pinecone_index_name=idx,
        tavily_api_key=tv,
        gemini_chat_model=_normalize_gemini_chat_model(
            os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash")
        ),
        gemini_embedding_model=os.getenv(
            "GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"
        ).strip(),
        use_multi_query=_truthy(os.getenv("USE_MULTI_QUERY"), False),
        max_upload_files=int(os.getenv("MAX_UPLOAD_FILES", "5")),
        max_pages_per_file=int(os.getenv("MAX_PAGES_PER_FILE", "10")),
        retrieval_k=int(os.getenv("RETRIEVAL_K", "5")),
        use_rerank=_truthy(os.getenv("USE_RERANK"), True),
        retrieval_fetch_multiplier=max(
            1, min(10, int(os.getenv("RETRIEVAL_FETCH_MULTIPLIER", "4")))
        ),
        pinecone_auto_create_index=_truthy(
            os.getenv("PINECONE_AUTO_CREATE_INDEX"), True
        ),
        pinecone_cloud=os.getenv("PINECONE_CLOUD", "aws").strip(),
        pinecone_region=os.getenv("PINECONE_REGION", "us-east-1").strip(),
        pinecone_embedding_dimension=int(
            os.getenv("PINECONE_EMBEDDING_DIMENSION", "768")
        ),
        pinecone_metric=os.getenv("PINECONE_METRIC", "cosine").strip(),
        agent_max_iterations=max(1, int(os.getenv("AGENT_MAX_ITERATIONS", "5"))),
        gemini_max_retries=max(
            0, min(8, int(os.getenv("GEMINI_MAX_RETRIES", "1")))
        ),
        chunk_size=int(os.getenv("CHUNK_SIZE", "1200")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "180")),
        agent_system_prompt=agent_system_prompt,
        tavily_tool_description=tavily_desc,
        document_search_tool_description=doc_tool_desc,
    )
