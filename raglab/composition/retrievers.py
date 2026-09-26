from langchain_core.retrievers import BaseRetriever

from raglab.composition.config import VectorRetrieverConfig
from raglab.composition.embeddings import build_embeddings
from raglab.composition.vector_stores import build_vector_store


def build_vector_retriever(
    config: VectorRetrieverConfig,
) -> BaseRetriever:
    """Build a vector retriever from configuration."""

    embeddings = build_embeddings(config.embedding)
    vector_store = build_vector_store(config.vector_store, embeddings)

    return vector_store.as_retriever(
        search_kwargs={"k": config.policy.top_k},
    )
