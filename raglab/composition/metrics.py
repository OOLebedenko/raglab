from functools import partial
from typing import assert_never

from raglab.application.evaluation.metrics import (
    Metric,
    recall_at_k,
    reciprocal_rank,
)
from raglab.composition.config import (
    MetricConfig,
    RecallAtKConfig,
    ReciprocalRankConfig,
)


def _build_metric(config: MetricConfig) -> Metric:
    """Build a single evaluation metric."""

    if isinstance(config, RecallAtKConfig):
        return Metric(
            name=f"recall@{config.k}",
            calculate=partial(recall_at_k, k=config.k),
        )

    if isinstance(config, ReciprocalRankConfig):
        return Metric(
            name="mrr",
            calculate=reciprocal_rank,
        )

    assert_never(config)


def build_metrics(configs: tuple[MetricConfig, ...]) -> list[Metric]:
    """Build evaluation metrics from configuration."""

    return [_build_metric(config) for config in configs]
