"""LangChain embeddings + Pinecone vector store (namespace = session)."""

from __future__ import annotations

import logging
import time

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from core.config import Settings

logger = logging.getLogger(__name__)

_INDEX_READY_TIMEOUT_SECONDS = 300
_INDEX_POLL_INTERVAL_SECONDS = 2


def ensure_pinecone_index(settings: Settings) -> None:
    """
    Ensure the configured index exists. If it does not:
    - When PINECONE_AUTO_CREATE_INDEX is true (default), create a serverless index
      (dimension must match ``GoogleGenerativeAIEmbeddings`` output, default 768).
    - Otherwise raise with instructions for the Pinecone console.
    """
    pc = Pinecone(api_key=settings.pinecone_api_key)
    name = settings.pinecone_index_name
    if pc.has_index(name):
        return

    if not settings.pinecone_auto_create_index:
        raise ValueError(
            f"Pinecone index '{name}' was not found (404). "
            f"Create a serverless index with dimension {settings.pinecone_embedding_dimension} "
            f"and metric {settings.pinecone_metric} in the Pinecone console, or set PINECONE_AUTO_CREATE_INDEX=true in .env."
        )

    logger.info(
        "Creating Pinecone index %r (%s dims, cosine, %s/%s)",
        name,
        settings.pinecone_embedding_dimension,
        settings.pinecone_cloud,
        settings.pinecone_region,
    )
    pc.create_index(
        name=name,
        dimension=settings.pinecone_embedding_dimension,
        metric=settings.pinecone_metric,
        spec=ServerlessSpec(
            cloud=settings.pinecone_cloud,
            region=settings.pinecone_region,
        ),
    )
    deadline = time.time() + _INDEX_READY_TIMEOUT_SECONDS
    poll = _INDEX_POLL_INTERVAL_SECONDS
    while time.time() < deadline:
        desc = pc.describe_index(name)
        if desc.status.ready:
            logger.info("Pinecone index %r is ready.", name)
            return
        time.sleep(poll)
    raise TimeoutError(
        f"Pinecone index {name!r} did not become ready within "
        f"{_INDEX_READY_TIMEOUT_SECONDS} seconds. Check the Pinecone dashboard."
    )


def get_embeddings(settings: Settings) -> GoogleGenerativeAIEmbeddings:
    """Use Gemini embedding model + output_dimensionality aligned with the Pinecone index."""
    return GoogleGenerativeAIEmbeddings(
        model=settings.gemini_embedding_model,
        google_api_key=settings.google_api_key,
        output_dimensionality=settings.pinecone_embedding_dimension,
    )


def get_vectorstore(
    *,
    settings: Settings,
    session_id: str,
) -> PineconeVectorStore:
    """Return a vector store scoped to this Streamlit session (Pinecone namespace)."""
    ensure_pinecone_index(settings)
    embeddings = get_embeddings(settings)
    return PineconeVectorStore(
        index_name=settings.pinecone_index_name,
        embedding=embeddings,
        namespace=session_id,
        pinecone_api_key=settings.pinecone_api_key,
    )


def add_documents_to_session(
    vectorstore: PineconeVectorStore,
    documents: list,
) -> int:
    """Upsert chunked documents into the session namespace."""
    if not documents:
        return 0
    vectorstore.add_documents(documents)
    return len(documents)
