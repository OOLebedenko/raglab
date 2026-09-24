import pytest

from raglab.application.evaluation.metrics import Metric
from raglab.application.evaluation_service import EvaluationService
from raglab.application.models import (
    EvaluationSample,
    RetrievedChunk,
)
from tests.unit.application.fakes import (
    FakeRelevanceJudge,
    FakeRetriever,
)


def test_evaluate_returns_aggregated_metrics(
    retrieved_chunks: list[RetrievedChunk],
    evaluation_samples: list[EvaluationSample],
) -> None:
    relevance_matches = frozenset(
        {
            (0, 0),
        }
    )

    metrics = [
        Metric(
            name="gold_count",
            calculate=lambda query_index: float(query_index.gold_count),
        ),
        Metric(
            name="match_count",
            calculate=lambda query_index: float(len(query_index.matches)),
        ),
    ]

    retriever = FakeRetriever(retrieved_chunks)
    relevance_judge = FakeRelevanceJudge(relevance_matches)

    service = EvaluationService(
        retriever=retriever,
        relevance_judge=relevance_judge,
    )

    result = service.evaluate(
        samples=evaluation_samples,
        metrics=metrics,
    )

    expected_gold_count = sum(
        len(sample.supporting_facts) for sample in evaluation_samples
    ) / len(evaluation_samples)
    expected_match_count = float(len(relevance_matches))

    assert result["gold_count"] == pytest.approx(expected_gold_count)
    assert result["match_count"] == pytest.approx(expected_match_count)
