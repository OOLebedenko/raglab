from pathlib import Path
from typing import Literal

from pydantic import PositiveInt

from raglab.composition.config.base import ConfigModel


class RetrievalPolicyConfig(ConfigModel):
    """Configure retrieval behavior."""

    top_k: PositiveInt


class EmbeddingConfig(ConfigModel):
    """Configure an embedding implementation."""

    type: str
    model_name: str


class VectorStoreConfig(ConfigModel):
    """Configure a vector store implementation."""

    type: str
    collection_name: str | None = None
    persist_directory: Path | None = None


class VectorRetrieverConfig(ConfigModel):
    """Configure vector retrieval."""

    type: Literal["vector"]
    policy: RetrievalPolicyConfig
    embedding: EmbeddingConfig
    vector_store: VectorStoreConfig


class LexicalRetrieverConfig(ConfigModel):
    """Configure lexical retrieval."""

    type: Literal["lexical"]
    implementation: str
    policy: RetrievalPolicyConfig
