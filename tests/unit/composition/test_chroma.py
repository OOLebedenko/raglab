from pathlib import Path
from typing import cast
from unittest.mock import Mock

import pytest
from langchain_core.embeddings import Embeddings

from raglab.infrastructure.indexes.chroma import load_chroma


@pytest.fixture
def embeddings() -> Embeddings:
    """Provide a mock embedding model."""

    return cast(Embeddings, Mock(spec=Embeddings))


def test_load_chroma_requires_existing_directory(
    tmp_path: Path,
    embeddings: Embeddings,
) -> None:
    """Reject a missing index directory."""

    with pytest.raises(FileNotFoundError, match="Chroma index directory"):
        load_chroma(
            tmp_path / "missing",
            embeddings,
            {"collection_name": "test_collection"},
        )


def test_load_chroma_requires_collection_name(
    tmp_path: Path,
    embeddings: Embeddings,
) -> None:
    """Reject Chroma configuration without a collection name."""

    with pytest.raises(ValueError, match="requires a collection_name"):
        load_chroma(tmp_path, embeddings, {})
