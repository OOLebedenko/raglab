from collections.abc import Callable
from pathlib import Path
from typing import assert_never

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
from raglab.composition.vector_stores import build_vector_store
from raglab.infrastructure.retrieval.adapter import LangChainRetrieverAdapter
from raglab.infrastructure.retrieval.lexical import BM25Retriever

_LEXICAL_BUILDERS: dict[
    str,
    Callable[[Path, int], BaseRetriever],
] = {
    "bm25s": lambda path, k: BM25Retriever.from_index(
        path,
        k=k,
    ),
}


def build_vector_retriever(
    config: VectorRetrieverConfig,
    policy: RetrievalPolicyConfig,
) -> BaseRetriever:
    """Build a vector retriever from configuration."""

    if not isinstance(config.index, LocalIndexConfig):
        raise NotImplementedError("Remote indexes are not implemented")

    embeddings = build_embeddings(config.embedding)
    vector_store = build_vector_store(config.index, embeddings)

    return vector_store.as_retriever(
        search_kwargs={"k": policy.top_k},
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
