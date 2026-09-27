from collections.abc import Callable
from pathlib import Path
from typing import Any, assert_never

from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever

from raglab.application.ports.retriever import Retriever
from raglab.composition.config import (
    LexicalRetrieverConfig,
    LocalIndexConfig,
    RetrievalPolicyConfig,
    RetrieverConfig,
    VectorRetrieverConfig,
)
from raglab.composition.embeddings import build_embeddings
from raglab.infrastructure.indexes.bm25 import load_bm25_index
from raglab.infrastructure.indexes.chroma import load_chroma
from raglab.infrastructure.retrieval.adapter import LangChainRetrieverAdapter
from raglab.infrastructure.retrieval.lexical import BM25Retriever


def _build_bm25_retriever(path: Path, k: int) -> BaseRetriever:
    """Build a retriever from an existing BM25S index."""

    index, tokenizer = load_bm25_index(path)

    return BM25Retriever(
        index=index,
        tokenizer=tokenizer,
        k=k,
    )


_LEXICAL_BUILDERS: dict[
    str,
    Callable[[Path, int], BaseRetriever],
] = {
    "bm25s": _build_bm25_retriever,
}

_VECTOR_BUILDERS: dict[
    str,
    Callable[[Path, Embeddings, dict[str, Any], int], BaseRetriever],
] = {
    "chroma": lambda path, embeddings, options, k: load_chroma(
        path,
        embeddings,
        options,
    ).as_retriever(search_kwargs={"k": k}),
}


def build_vector_retriever(
    config: VectorRetrieverConfig,
    policy: RetrievalPolicyConfig,
) -> BaseRetriever:
    """Build a vector retriever from configuration."""

    if not isinstance(config.index, LocalIndexConfig):
        raise NotImplementedError("Remote indexes are not implemented")

    try:
        builder = _VECTOR_BUILDERS[config.index.type]
    except KeyError as exc:
        raise ValueError(f"Unsupported vector retriever: {config.index.type}") from exc

    embeddings = build_embeddings(config.embedding)

    return builder(
        config.index.path,
        embeddings,
        config.index.options,
        policy.top_k,
    )


def build_lexical_retriever(
    config: LexicalRetrieverConfig,
    policy: RetrievalPolicyConfig,
) -> BaseRetriever:
    """Build a lexical retriever from configuration."""

    if not isinstance(config.index, LocalIndexConfig):
        raise NotImplementedError("Remote indexes are not implemented")

    try:
        builder = _LEXICAL_BUILDERS[config.index.type]
    except KeyError as exc:
        raise ValueError(f"Unsupported lexical retriever: {config.index.type}") from exc

    return builder(config.index.path, policy.top_k)


def build_retriever(
    config: RetrieverConfig,
    policy: RetrievalPolicyConfig,
) -> Retriever:
    """Build and adapt a retriever from configuration."""

    if isinstance(config, VectorRetrieverConfig):
        retriever = build_vector_retriever(config, policy)

    elif isinstance(config, LexicalRetrieverConfig):
        retriever = build_lexical_retriever(config, policy)

    else:
        assert_never(config)

    return LangChainRetrieverAdapter(retriever=retriever)
