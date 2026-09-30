from pathlib import Path
from typing import Annotated

from pydantic import Field

from raglab.composition.config.base import ConfigModel
from raglab.composition.config.evaluation import (
    LlmJudgeConfig,
    RecallAtKConfig,
    ReciprocalRankConfig,
    SubstringJudgeConfig,
)
from raglab.composition.config.generation import GeneratorConfig
from raglab.composition.config.retrieval import (
    LexicalRetrieverConfig,
    RetrievalPolicyConfig,
    VectorRetrieverConfig,
)

type RetrieverConfig = Annotated[
    VectorRetrieverConfig | LexicalRetrieverConfig,
    Field(discriminator="type"),
]

type JudgeConfig = Annotated[
    SubstringJudgeConfig | LlmJudgeConfig,
    Field(discriminator="type"),
]

type MetricConfig = Annotated[
    RecallAtKConfig | ReciprocalRankConfig,
    Field(discriminator="type"),
]


class DataConfig(ConfigModel):
    """Configure paths to canonical RAGLab data."""

    chunks: Path
    benchmark: Path


class EvaluationConfig(ConfigModel):
    """Configure retrieval evaluation."""

    judge: JudgeConfig
    metrics: tuple[MetricConfig, ...] = Field(min_length=1)


class ExperimentConfig(ConfigModel):
    """Configure a RAG experiment."""

    data: DataConfig
    retriever: RetrieverConfig
    retrieval_policy: RetrievalPolicyConfig
    evaluation: EvaluationConfig
    generator: GeneratorConfig | None = None


class ProductionConfig(ConfigModel):
    """Configure a RAG pipeline for serving queries."""

    retriever: RetrieverConfig
    retrieval_policy: RetrievalPolicyConfig
    generator: GeneratorConfig
