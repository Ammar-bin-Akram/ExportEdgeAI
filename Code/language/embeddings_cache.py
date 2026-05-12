"""Shared cached embedding model loader for language modules."""

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cpu"


@lru_cache(maxsize=1)
def get_cached_embeddings() -> HuggingFaceEmbeddings:
    """Return a process-wide cached HuggingFace embedding model instance."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": EMBEDDING_DEVICE},
    )
