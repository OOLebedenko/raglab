from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from raglab.composition.config import (
    ExperimentConfig,
    LexicalRetrieverConfig,
    LocalIndexConfig,
    VectorRetrieverConfig,
    load_config,
)


@pytest.fixture
def config_data() -> dict[str, Any]:
    """Provide a valid experiment configuration."""

    return {
        "data": {
            "chunks": "data/chunks.json",
            "benchmark": "data/benchmark.json",
        },
        "retriever": {
            "type": "vector",
            "embedding": {
                "type": "huggingface",
                "model_name": "BAAI/bge-m3",
            },
            "index": {
                "type": "chroma",
                "location": "local",
                "path": "data/chroma",
                "options": {
                    "collection_name": "raglab",
                },
            },
        },
        "retrieval_policy": {
            "top_k": 10,
        },
        "generator": {
            "model": {
                "provider": "gigachat",
                "name": "GigaChat-2-Max",
            },
        },
        "evaluation": {
            "judge": {
                "type": "substring",
            },
            "metrics": [
                {
                    "type": "recall_at_k",
                    "k": 1,
                },
                {
                    "type": "reciprocal_rank",
                },
            ],
        },
    }


@pytest.fixture
def lexical_retriever_data() -> dict[str, Any]:
    """Provide a valid lexical retriever configuration."""

    return {
        "type": "lexical",
        "index": {
            "type": "bm25s",
            "location": "local",
            "path": "data/bm25",
        },
    }


def test_load_config_resolves_relative_paths(
    tmp_path: Path,
    config_data: dict[str, Any],
) -> None:
    """Resolve relative paths from the project root."""

    config_dir = tmp_path / "configs"
    config_dir.mkdir()

    config_path = config_dir / "experiment.yaml"
    config_path.write_text(
        yaml.safe_dump(config_data),
        encoding="utf-8",
    )

    config = load_config(
        path=Path("configs/experiment.yaml"),
        project_root=tmp_path,
    )

    assert config.data.chunks == tmp_path / "data/chunks.json"
    assert config.data.benchmark == tmp_path / "data/benchmark.json"

    assert isinstance(config.retriever, VectorRetrieverConfig)
    assert isinstance(config.retriever.index, LocalIndexConfig)
    assert config.retriever.index.path == tmp_path / "data/chroma"


def test_load_lexical_config(
    tmp_path: Path,
    config_data: dict[str, Any],
    lexical_retriever_data: dict[str, Any],
) -> None:
    """Load a lexical retriever without vector settings."""

    config_data["retriever"] = lexical_retriever_data
    config_data["retrieval_policy"] = {"top_k": 5}

    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        yaml.safe_dump(config_data),
        encoding="utf-8",
    )

    config = load_config(
        path=config_path,
        project_root=tmp_path,
    )

    assert isinstance(config.retriever, LexicalRetrieverConfig)
    assert isinstance(config.retriever.index, LocalIndexConfig)
    assert config.retriever.index.type == "bm25s"
    assert config.retriever.index.path == tmp_path / "data/bm25"
    assert config.retrieval_policy.top_k == 5
    assert config.data.chunks == tmp_path / "data/chunks.json"


def test_vector_retriever_requires_embedding(
    config_data: dict[str, Any],
) -> None:
    """Reject vector retrieval without embedding configuration."""

    del config_data["retriever"]["embedding"]

    with pytest.raises(ValidationError, match="embedding"):
        ExperimentConfig.model_validate(config_data)


def test_lexical_retriever_rejects_vector_settings(
    config_data: dict[str, Any],
    lexical_retriever_data: dict[str, Any],
) -> None:
    """Reject vector-specific settings for lexical retrieval."""

    lexical_retriever_data["embedding"] = config_data["retriever"]["embedding"]
    config_data["retriever"] = lexical_retriever_data

    with pytest.raises(ValidationError, match="embedding"):
        ExperimentConfig.model_validate(config_data)


def test_experiment_config_without_generator(
    config_data: dict[str, Any],
) -> None:
    """Allow retrieval evaluation without answer generation."""

    del config_data["generator"]

    config = ExperimentConfig.model_validate(config_data)

    assert config.generator is None


def test_experiment_config_with_generator(
    config_data: dict[str, Any],
) -> None:
    """Allow experiments with answer generation."""

    config = ExperimentConfig.model_validate(config_data)

    assert config.generator is not None
    assert config.generator.model.provider == "gigachat"


def test_config_is_immutable(
    config_data: dict[str, Any],
) -> None:
    """Prevent changes to validated configuration."""

    config = ExperimentConfig.model_validate(config_data)

    with pytest.raises(ValidationError, match="frozen"):
        config.data = config.data

    with pytest.raises(ValidationError, match="frozen"):
        config.data.chunks = Path("other.json")

    assert isinstance(config.evaluation.metrics, tuple)
