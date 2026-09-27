from raglab.application.models import QueryRelevance
from raglab.composition.config import (
    RecallAtKConfig,
    ReciprocalRankConfig,
)
from raglab.composition.metrics import build_metrics


def test_build_metrics() -> None:
    """Build configured evaluation metrics."""

    configs = (
        RecallAtKConfig(type="recall_at_k", k=1),
        ReciprocalRankConfig(type="reciprocal_rank"),
    )

    metrics = build_metrics(configs)

    # Two gold facts are found at ranks 1 and 3.
    index = QueryRelevance(
        matches=frozenset({(0, 0), (2, 1)}),
        gold_count=2,
    )

    # Metric names follow the configured order.
    assert [metric.name for metric in metrics] == ["recall@1", "mrr"]

    # Each metric computes against the same index.
    assert [metric.calculate(index) for metric in metrics] == [0.5, 1.0]


def test_build_metrics_binds_recall_cutoffs() -> None:
    """Bind a separate cutoff to each recall metric."""

    configs = (
        RecallAtKConfig(type="recall_at_k", k=1),
        RecallAtKConfig(type="recall_at_k", k=3),
    )

    metrics = build_metrics(configs)

    # The only gold fact is found at rank 3.
    index = QueryRelevance(
        matches=frozenset({(2, 0)}),
        gold_count=1,
    )

    # Each recall metric keeps its own cutoff, not the last one.
    assert [metric.name for metric in metrics] == ["recall@1", "recall@3"]

    # The single match at rank 3 is outside recall@1 and inside recall@3.
    assert [metric.calculate(index) for metric in metrics] == [0.0, 1.0]
