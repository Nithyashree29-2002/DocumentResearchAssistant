"""PDF loading, validation, and LangChain chunking."""

from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from core.config import Settings

_CHUNK_SEPARATORS = ("\n\n", "\n", ". ", " ", "")


def _count_pages(path: Path) -> int:
    reader = PdfReader(str(path))
    return len(reader.pages)


def load_and_chunk_pdfs(
    *,
    uploaded_paths: list[Path],
    session_id: str,
    settings: Settings,
    source_filenames: list[str] | None = None,
) -> list[Document]:
    """Load PDFs, enforce limits, split with metadata for citations.

    ``source_filenames``: original upload names for citations (same order as ``uploaded_paths``).
    When omitted, ``path.name`` is used (e.g. temp files — pass names from the UI).
    """
    if len(uploaded_paths) > settings.max_upload_files:
        raise ValueError(
            f"At most {settings.max_upload_files} PDFs allowed; got {len(uploaded_paths)}."
        )
    if source_filenames is not None and len(source_filenames) != len(uploaded_paths):
        raise ValueError(
            "source_filenames must be the same length as uploaded_paths."
        )

    all_docs: list[Document] = []
    for i, path in enumerate(uploaded_paths):
        display_name = (
            source_filenames[i] if source_filenames else path.name
        )
        pages = _count_pages(path)
        if pages > settings.max_pages_per_file:
            raise ValueError(
                f"{display_name} has {pages} pages; max {settings.max_pages_per_file} per file."
            )
        loader = PyPDFLoader(str(path))
        pages_docs = loader.load()
        for d in pages_docs:
            d.metadata["source"] = Path(display_name).name
            d.metadata["session_id"] = session_id
        all_docs.extend(pages_docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=list(_CHUNK_SEPARATORS),
    )
    splits = splitter.split_documents(all_docs)
    for i, doc in enumerate(splits):
        doc.metadata["chunk_index"] = i
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        if page is not None:
            doc.metadata["citation_label"] = f"{src} — p. {int(page) + 1}"
        else:
            doc.metadata["citation_label"] = src
    return splits
