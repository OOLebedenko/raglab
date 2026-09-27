from pathlib import Path
from typing import cast
from unittest.mock import Mock

import pytest
from langchain_core.embeddings import Embeddings

from raglab.composition.config import LocalIndexConfig
from raglab.composition.vector_stores import build_vector_store


@pytest.fixture
def embeddings() -> Embeddings:
    """Provide a mock embedding model."""

    return cast(Embeddings, Mock(spec=Embeddings))


def test_build_chroma_store_requires_existing_directory(
    tmp_path: Path,
    embeddings: Embeddings,
) -> None:
    """Reject a missing index directory."""

    config = LocalIndexConfig(
        type="chroma",
        location="local",
        path=tmp_path / "missing",
        options={"collection_name": "test_collection"},
    )

    with pytest.raises(FileNotFoundError, match="Chroma index directory"):
        build_vector_store(config, embeddings)


def test_build_chroma_store_requires_collection_name(
    tmp_path: Path,
    embeddings: Embeddings,
) -> None:
    """Reject Chroma configuration without a collection name."""

    config = LocalIndexConfig(
        type="chroma",
        location="local",
        path=tmp_path,
    )

    with pytest.raises(ValueError, match="requires a collection_name"):
        build_vector_store(config, embeddings)


def test_build_vector_store_rejects_unknown_type(
    tmp_path: Path,
    embeddings: Embeddings,
) -> None:
    """Reject an unsupported vector store implementation."""

    config = LocalIndexConfig(
        type="unknown",
        location="local",
        path=tmp_path,
    )

    with pytest.raises(ValueError, match="Unsupported vector store"):
        build_vector_store(config, embeddings)
