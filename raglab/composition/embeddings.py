from collections.abc import Callable

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from raglab.composition.config import EmbeddingConfig

# Currently only HuggingFace is supported.
# Add new embedding providers by registering their builders here.
_EMBEDDING_BUILDERS: dict[str, Callable[[str], Embeddings]] = {
    "huggingface": lambda name: HuggingFaceEmbeddings(model_name=name),
}


def build_embeddings(config: EmbeddingConfig) -> Embeddings:
    """Build an embedding model from configuration."""

    try:
        builder = _EMBEDDING_BUILDERS[config.type]
    except KeyError as exc:
        raise ValueError(f"Unsupported embedding provider: {config.type}") from exc

    return builder(config.model_name)
