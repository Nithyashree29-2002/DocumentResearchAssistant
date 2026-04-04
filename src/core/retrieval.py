"""LangChain retrievers: vector search, optional metadata filter, FlashRank rerank, optional multi-query."""

from __future__ import annotations

import logging
from typing import Any

from flashrank.Ranker import RerankRequest
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from pydantic import ConfigDict, Field

from core.config import Settings

logger = logging.getLogger(__name__)

_RETRIEVAL_LLM_TEMPERATURE = 0.0
_RETRIEVAL_LLM_STREAMING = False

# Lazy singleton — model download on first use (~3 MB default).
_ranker: Any = None


def _get_flashrank_ranker() -> Any:
    global _ranker
    if _ranker is None:
        from flashrank import Ranker

        _ranker = Ranker()
        logger.info("FlashRank cross-encoder loaded for retrieval reranking.")
    return _ranker


class FlashrankRerankingRetriever(BaseRetriever):
    """Retrieve a wide candidate set, rerank with a local cross-encoder, return top-N."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    base_retriever: BaseRetriever = Field(repr=False)
    top_n: int = Field(ge=1, le=50)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        child = run_manager.get_child() if run_manager else None
        cfg = {"callbacks": child} if child is not None else {}
        docs = list(self.base_retriever.invoke(query, config=cfg))
        if not docs:
            return []
        if len(docs) <= self.top_n:
            return docs

        ranker = _get_flashrank_ranker()
        passages = [
            {"id": str(i), "text": (d.page_content or "")[:12000]}
            for i, d in enumerate(docs)
        ]
        ranked = ranker.rerank(RerankRequest(query=query, passages=passages))
        out: list[Document] = []
        for row in ranked[: self.top_n]:
            try:
                idx = int(row["id"])
            except (KeyError, ValueError, TypeError):
                continue
            if 0 <= idx < len(docs):
                out.append(docs[idx])
        return out if out else docs[: self.top_n]


def build_retriever(
    *,
    vectorstore: PineconeVectorStore,
    settings: Settings,
    metadata_filter: dict | None = None,
) -> BaseRetriever:
    k_out = max(1, settings.retrieval_k)
    if settings.use_rerank:
        fetch_k = min(50, max(k_out, k_out * max(1, settings.retrieval_fetch_multiplier)))
    else:
        fetch_k = k_out

    search_kwargs: dict[str, Any] = {"k": fetch_k}
    if metadata_filter:
        search_kwargs["filter"] = metadata_filter

    base_vs = vectorstore.as_retriever(search_kwargs=search_kwargs)

    if settings.use_rerank and fetch_k > k_out:
        inner: BaseRetriever = FlashrankRerankingRetriever(
            base_retriever=base_vs,
            top_n=k_out,
        )
    else:
        inner = base_vs

    if not settings.use_multi_query:
        return inner

    llm = ChatGoogleGenerativeAI(
        model=settings.gemini_chat_model,
        google_api_key=settings.google_api_key,
        temperature=_RETRIEVAL_LLM_TEMPERATURE,
        max_retries=settings.gemini_max_retries,
        streaming=_RETRIEVAL_LLM_STREAMING,
    )
    return MultiQueryRetriever.from_llm(
        retriever=inner,
        llm=llm,
        include_original=True,
    )
