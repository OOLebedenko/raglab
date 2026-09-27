from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, PositiveInt

from raglab.composition.config.base import ConfigModel


class RetrievalPolicyConfig(ConfigModel):
    """Configure retrieval behavior."""

    top_k: PositiveInt


class EmbeddingConfig(ConfigModel):
    """Configure an embedding implementation."""

    type: str
    model_name: str


class IndexConfig(ConfigModel):
    """Base configuration for a search index.

    Options contain non-sensitive implementation-specific parameters.
    Secrets must be provided separately through environment settings.
    """

    type: str
    options: dict[str, Any] = Field(default_factory=dict)


class LocalIndexConfig(IndexConfig):
    """Configure an existing local search index.

    The path points to a prepared index. Its interpretation
    depends on the index implementation.
    """

    location: Literal["local"]
    path: Path


class RemoteIndexConfig(IndexConfig):
    """Configure a remote search index."""

    location: Literal["remote"]
    endpoint: str = Field(min_length=1)


type IndexConfigVariant = Annotated[
    LocalIndexConfig | RemoteIndexConfig,
    Field(discriminator="location"),
]


class VectorRetrieverConfig(ConfigModel):
    """Configure vector retrieval."""

    type: Literal["vector"]
    policy: RetrievalPolicyConfig
    embedding: EmbeddingConfig
    index: IndexConfigVariant


class LexicalRetrieverConfig(ConfigModel):
    """Configure lexical retrieval."""

    type: Literal["lexical"]
    policy: RetrievalPolicyConfig
    index: IndexConfigVariant
