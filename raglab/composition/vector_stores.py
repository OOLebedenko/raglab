from collections.abc import Callable
from pathlib import Path
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from raglab.composition.config import LocalIndexConfig
from raglab.infrastructure.vector_stores.chroma import load_chroma

# Register loaders for additional vector stores here.
_STORE_LOADERS: dict[
    str,
    Callable[[Path, Embeddings, dict[str, Any]], VectorStore],
] = {
    "chroma": load_chroma,
}


def build_vector_store(
    config: LocalIndexConfig,
    embeddings: Embeddings,
) -> VectorStore:
    """Connect to an existing local vector store."""

    try:
        loader = _STORE_LOADERS[config.type]
    except KeyError as exc:
        raise ValueError(f"Unsupported vector store: {config.type}") from exc

    return loader(config.path, embeddings, config.options)
