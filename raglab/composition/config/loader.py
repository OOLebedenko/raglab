from pathlib import Path

import yaml

from raglab.composition.config.retrieval import LocalIndexConfig
from raglab.composition.config.schema import (
    ExperimentConfig,
    ProductionConfig,
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
    """Resolve the path of a local search index."""

    index = retriever.index

    if not isinstance(index, LocalIndexConfig):
        return retriever

    resolved_index = index.model_copy(
        update={
            "path": resolve_path(index.path, project_root),
        }
    )

    return retriever.model_copy(update={"index": resolved_index})


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


def load_production_config(
    path: Path,
    project_root: Path,
) -> ProductionConfig:
    """Load production configuration and resolve its index path."""

    root = project_root.resolve()
    config_path = resolve_path(path, root)

    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = ProductionConfig.model_validate(data)

    return config.model_copy(
        update={
            "retriever": _resolve_retriever_paths(
                config.retriever,
                root,
            )
        }
    )
