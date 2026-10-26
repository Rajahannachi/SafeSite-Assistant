"""Ingestion: load documents from data/, split them, embed them, persist them in Chroma.

Re-running the script is idempotent: every chunk gets a deterministic id
(a hash of its file, page, position and text), so unchanged chunks are skipped,
new or modified chunks are added and chunks that disappeared are deleted.

Usage:
    python ingest.py            # incremental sync
    python ingest.py --reset    # drop the collection and rebuild it
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import get_settings
from store import get_vectorstore

logger = logging.getLogger("ingest")

SUPPORTED_SUFFIXES = {".pdf", ".md", ".txt"}


def load_documents(data_dir: Path) -> list[Document]:
    """Load every supported file; one Document per PDF page, one per text file.

    Metadata is normalised to ``source`` (file name, used in citations) and
    ``page`` (1-based; text and Markdown files have a single page).
    """
    docs: list[Document] = []
    for path in sorted(data_dir.iterdir()):
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            continue
        if suffix == ".pdf":
            pages = PyPDFLoader(str(path)).load()
            for page in pages:
                page.metadata = {"source": path.name, "page": int(page.metadata.get("page", 0)) + 1}
            docs.extend(pages)
        else:
            text_doc = TextLoader(str(path), encoding="utf-8").load()[0]
            text_doc.metadata = {"source": path.name, "page": 1}
            docs.append(text_doc)
        logger.info("loaded %s", path.name)
    return docs


def split_documents(docs: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Split pages into overlapping chunks.

    The separators favour Markdown headings and blank lines first so that a
    chunk tends to hold one complete rule or section.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n\n", "\n", ". ", " ", ""],
        add_start_index=True,  # character offset of the chunk inside its page
    )
    return splitter.split_documents(docs)


def chunk_id(chunk: Document) -> str:
    """Deterministic id: identical chunk -> identical id -> no duplicate on re-run."""
    key = f"{chunk.metadata['source']}|{chunk.metadata['page']}|{chunk.metadata['start_index']}|{chunk.page_content}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]


def sync_vectorstore(chunks: list[Document], reset: bool = False) -> dict[str, int]:
    """Add new chunks and delete stale ones so that the store mirrors data/."""
    store = get_vectorstore()
    if reset:
        store.reset_collection()

    wanted = {chunk_id(c): c for c in chunks}  # dict also removes exact duplicates
    existing = set(store.get(include=[])["ids"])

    to_add = [cid for cid in wanted if cid not in existing]
    to_delete = [cid for cid in existing if cid not in wanted]

    if to_delete:
        store.delete(ids=to_delete)
    if to_add:
        store.add_documents([wanted[cid] for cid in to_add], ids=to_add)

    return {"chunks": len(wanted), "added": len(to_add), "deleted": len(to_delete),
            "unchanged": len(wanted) - len(to_add)}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Ingest safety documents into the vector store.")
    parser.add_argument("--reset", action="store_true", help="drop and rebuild the collection")
    args = parser.parse_args()

    settings = get_settings()
    docs = load_documents(settings.data_dir)
    if not docs:
        raise SystemExit(f"No documents found in {settings.data_dir}")

    chunks = split_documents(docs, settings.chunk_size, settings.chunk_overlap)
    stats = sync_vectorstore(chunks, reset=args.reset)
    logger.info("%d files, %d pages -> %s", len({d.metadata['source'] for d in docs}), len(docs), stats)


if __name__ == "__main__":
    main()
