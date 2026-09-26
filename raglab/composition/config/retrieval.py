from pathlib import Path
from typing import Any, Literal

from pydantic import Field, PositiveInt

from raglab.composition.config.base import ConfigModel


class RetrievalPolicyConfig(ConfigModel):
    """Configure retrieval behavior."""

    top_k: PositiveInt


class EmbeddingConfig(ConfigModel):
    """Configure an embedding implementation."""

    type: str
    model_name: str


class LocalVectorStoreConfig(ConfigModel):
    """Configure an existing vector store.

    The path points to a prepared index. Its interpretation
    depends on the vector store implementation.

    Options contain non-sensitive implementation-specific parameters.
    Secrets must be provided separately through environment settings.
    """

    type: str
    path: Path
    options: dict[str, Any] = Field(default_factory=dict)


class VectorRetrieverConfig(ConfigModel):
    """Configure vector retrieval."""

    type: Literal["vector"]
    policy: RetrievalPolicyConfig
    embedding: EmbeddingConfig
    vector_store: LocalVectorStoreConfig


class LexicalRetrieverConfig(ConfigModel):
    """Configure lexical retrieval."""

    type: Literal["lexical"]
    implementation: str
    policy: RetrievalPolicyConfig
