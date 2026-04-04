"""Ingestion statistics over chunked documents (pre/post upsert)."""

from __future__ import annotations

import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from langchain_core.documents import Document


def _doc_source_basename(doc: Document) -> str:
    return Path(str(doc.metadata.get("source", "") or "")).name


def compute_ingestion_metrics(documents: list[Document]) -> dict[str, Any]:
    """Aggregate stats over chunked ``Document``s (before or after upsert)."""
    if not documents:
        return {
            "n_chunks": 0,
            "n_unique_sources": 0,
            "total_characters": 0,
            "chunk_length_mean": 0.0,
            "chunk_length_min": 0,
            "chunk_length_max": 0,
            "chunk_length_stdev": 0.0,
            "chunks_per_source": {},
            "pct_chunks_with_page": 0.0,
            "pct_chunks_with_citation_label": 0.0,
        }

    lengths = [len(d.page_content or "") for d in documents]
    sources = [_doc_source_basename(d) for d in documents]
    with_page = sum(1 for d in documents if d.metadata.get("page") is not None)
    with_cite = sum(
        1 for d in documents if str(d.metadata.get("citation_label", "")).strip()
    )
    per_src = Counter(sources)
    stdev = statistics.stdev(lengths) if len(lengths) > 1 else 0.0

    return {
        "n_chunks": len(documents),
        "n_unique_sources": len(per_src),
        "total_characters": sum(lengths),
        "chunk_length_mean": round(statistics.mean(lengths), 1),
        "chunk_length_min": min(lengths),
        "chunk_length_max": max(lengths),
        "chunk_length_stdev": round(stdev, 1),
        "chunks_per_source": dict(sorted(per_src.items(), key=lambda x: (-x[1], x[0]))),
        "pct_chunks_with_page": round(100.0 * with_page / len(documents), 1),
        "pct_chunks_with_citation_label": round(100.0 * with_cite / len(documents), 1),
    }
