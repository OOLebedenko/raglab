from pathlib import Path

import yaml

from raglab.composition.config.retrieval import VectorRetrieverConfig
from raglab.composition.config.schema import (
    ExperimentConfig,
    RetrieverConfig,
)


def resolve_path(
    path: Path,
    project_root: Path,
) -> Path:
    """Resolve a path relative to the project root."""

    if path.is_absolute():
        return path

    return project_root / path


def _resolve_retriever_paths(
    retriever: RetrieverConfig,
    project_root: Path,
) -> RetrieverConfig:
    """Resolve paths inside a retriever configuration."""

    if not isinstance(retriever, VectorRetrieverConfig):
        return retriever

    vector_store = retriever.vector_store

    resolved_store = vector_store.model_copy(
        update={
            "path": resolve_path(
                vector_store.path,
                project_root,
            ),
        }
    )

    return retriever.model_copy(update={"vector_store": resolved_store})


def load_config(
    path: Path,
    project_root: Path,
) -> ExperimentConfig:
    """Load and validate an experiment configuration."""

    resolved_root = project_root.resolve()
    config_path = resolve_path(path, resolved_root)

    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = ExperimentConfig.model_validate(data)

    resolved_data = config.data.model_copy(
        update={
            "chunks": resolve_path(config.data.chunks, resolved_root),
            "benchmark": resolve_path(config.data.benchmark, resolved_root),
        }
    )

    resolved_retriever = _resolve_retriever_paths(
        config.retriever,
        resolved_root,
    )

    return config.model_copy(
        update={
            "data": resolved_data,
            "retriever": resolved_retriever,
        }
    )
