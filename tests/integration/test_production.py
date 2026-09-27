from pathlib import Path

import pytest
import yaml
from langchain_core.documents import Document

from raglab.composition.config.loader import load_production_config
from raglab.composition.retrievers import build_retriever
from raglab.infrastructure.indexes.bm25 import build_bm25_index


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    """Create a temporary production configuration."""

    path = tmp_path / "production.yaml"

    config = {
        "retriever": {
            "type": "lexical",
            "index": {
                "type": "bm25s",
                "location": "local",
                "path": "bm25-production",
            },
        },
        "retrieval_policy": {"top_k": 1},
        "generator": {
            "model": {
                "provider": "gigachat",
                "name": "GigaChat-2-Max",
            },
        },
    }

    path.write_text(
        yaml.safe_dump(config),
        encoding="utf-8",
    )

    return path


@pytest.fixture
def prepared_bm25_index(tmp_path: Path) -> Path:
    """Prepare an index for the production retrieval test"""

    path = tmp_path / "bm25-production"

    documents = [
        Document(
            page_content="Quantum spin resonance in magnetic fields",
            metadata={"source": "physics.md"},
        ),
        Document(
            page_content="Database transactions and isolation levels",
            metadata={"source": "database.md"},
        ),
        Document(
            page_content="Web applications use HTTP for communication",
            metadata={"source": "web.md"},
        ),
    ]

    build_bm25_index(documents, path)

    return path


def test_production_retrieval_pipeline(
    config_path: Path,
    prepared_bm25_index: Path,
) -> None:
    """Load production configuration and retrieve from a prepared index."""

    # 1. Load the production configuration
    config = load_production_config(
        config_path,
        project_root=config_path.parent,
    )

    assert config.generator.model.provider == "gigachat"

    # 2. Connect to the prepared index using configuration
    retriever = build_retriever(
        config.retriever,
        config.retrieval_policy,
    )

    # 3. Perform retrieval
    retrieved = retriever.retrieve("quantum resonance")

    assert len(retrieved) == 1
    assert retrieved[0].text == "Quantum spin resonance in magnetic fields"
    assert retrieved[0].source == "physics.md"
