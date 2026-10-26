"""Shared access to the embedding model and the persisted Chroma vector store."""

from __future__ import annotations

from functools import lru_cache

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config import ROOT, get_settings


@lru_cache
def get_embeddings() -> HuggingFaceEmbeddings:
    """Local, free sentence-transformers model (downloaded once, then cached)."""
    settings = get_settings()
    # A copy in models/<name> (downloaded manually for offline installs) wins over the Hub.
    local_copy = ROOT / "models" / settings.embedding_model.split("/")[-1]
    return HuggingFaceEmbeddings(
        model_name=str(local_copy) if local_copy.is_dir() else settings.embedding_model,
        model_kwargs={"device": "cpu"},
        # Normalised vectors -> cosine similarity == dot product, scores in [0, 1].
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache
def get_vectorstore() -> Chroma:
    settings = get_settings()
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(settings.vectorstore_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )
