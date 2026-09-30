from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, assert_never

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from raglab.composition.config import (
    LexicalRetrieverConfig,
    LocalIndexConfig,
    RetrieverConfig,
    VectorRetrieverConfig,
)
from raglab.composition.embeddings import build_embeddings
from raglab.infrastructure.indexes.bm25 import build_bm25_index
from raglab.infrastructure.indexes.chroma import build_chroma_index

_LEXICAL_BUILDERS: dict[
    str,
    Callable[[Sequence[Document], Path], None],
] = {
    "bm25s": build_bm25_index,
}

_VECTOR_BUILDERS: dict[
    str,
    Callable[[Sequence[Document], Path, Embeddings, dict[str, Any]], None],
] = {
    "chroma": build_chroma_index,
}


def build_index(
    config: RetrieverConfig,
    documents: Sequence[Document],
) -> None:
    """Build an index using the configured implementation."""

    if not isinstance(config.index, LocalIndexConfig):
        raise NotImplementedError("Remote indexes are not implemented")

    if isinstance(config, LexicalRetrieverConfig):
        try:
            lexical_builder = _LEXICAL_BUILDERS[config.index.type]
        except KeyError as exc:
            raise ValueError(f"Unsupported lexical index: {config.index.type}") from exc

        lexical_builder(documents, config.index.path)

    elif isinstance(config, VectorRetrieverConfig):
        try:
            vector_builder = _VECTOR_BUILDERS[config.index.type]
        except KeyError as exc:
            raise ValueError(f"Unsupported vector index: {config.index.type}") from exc

        embeddings = build_embeddings(config.embedding)

        vector_builder(
            documents,
            config.index.path,
            embeddings,
            config.index.options,
        )
    else:
        assert_never(config)
