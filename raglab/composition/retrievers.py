from collections.abc import Callable, Sequence
from typing import assert_never

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from raglab.application.ports.retriever import Retriever
from raglab.composition.config import (
    LexicalRetrieverConfig,
    RetrieverConfig,
    VectorRetrieverConfig,
)
from raglab.composition.embeddings import build_embeddings
from raglab.composition.vector_stores import build_vector_store
from raglab.infrastructure.retrieval.adapter import LangChainRetrieverAdapter
from raglab.infrastructure.retrieval.lexical import BM25Retriever

_LEXICAL_BUILDERS: dict[
    str,
    Callable[[Sequence[Document], int], BaseRetriever],
] = {
    "bm25": lambda documents, k: BM25Retriever.from_documents(
        documents,
        k=k,
    ),
}


def build_vector_retriever(
    config: VectorRetrieverConfig,
) -> BaseRetriever:
    """Build a vector retriever from configuration."""

    embeddings = build_embeddings(config.embedding)
    vector_store = build_vector_store(config.vector_store, embeddings)

    return vector_store.as_retriever(
        search_kwargs={"k": config.policy.top_k},
    )


def build_lexical_retriever(
    config: LexicalRetrieverConfig,
    documents: Sequence[Document],
) -> BaseRetriever:
    """Build a lexical retriever from configuration."""

    try:
        builder = _LEXICAL_BUILDERS[config.implementation]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported lexical retriever: {config.implementation}"
        ) from exc

    return builder(documents, config.policy.top_k)


def build_retriever(
    config: RetrieverConfig,
    documents: Sequence[Document] | None = None,
) -> Retriever:
    """Build and adapt a retriever from configuration."""

    if isinstance(config, VectorRetrieverConfig):
        retriever = build_vector_retriever(config)

    elif isinstance(config, LexicalRetrieverConfig):
        if documents is None:
            raise ValueError("Documents are required for lexical retrieval")

        retriever = build_lexical_retriever(config, documents)

    else:
        assert_never(config)

    return LangChainRetrieverAdapter(retriever=retriever)
