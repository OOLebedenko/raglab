import pytest

from raglab.application.evaluation.metrics import Metric
from raglab.application.models import (
    EvaluationSample,
    EvaluationSuccess,
    RetrievedChunk,
)
from raglab.application.sample_evaluation_service import SampleEvaluationService
from tests.unit.application.fakes import (
    FakeRelevanceJudge,
    FakeRetriever,
)


def test_evaluate_sample_returns_metrics(
    retrieved_chunks: list[RetrievedChunk],
    evaluation_samples: list[EvaluationSample],
) -> None:
    relevance_matches = frozenset({(0, 0)})

    metrics = [
        Metric(
            name="gold_count",
            calculate=lambda relevance: float(relevance.gold_count),
        ),
        Metric(
            name="match_count",
            calculate=lambda relevance: float(len(relevance.matches)),
        ),
    ]

    retriever = FakeRetriever(retrieved_chunks)
    relevance_judge = FakeRelevanceJudge(relevance_matches)

    service = SampleEvaluationService(
        retriever=retriever,
        relevance_judge=relevance_judge,
        metrics=metrics,
    )

    for sample in evaluation_samples:
        result = service.evaluate_sample(sample)

        assert isinstance(result, EvaluationSuccess)
        assert result.sample == sample
        assert result.retrieved_chunks == retrieved_chunks

        assert result.relevance.matches == relevance_matches
        assert result.relevance.gold_count == len(sample.supporting_facts)

        assert result.metrics == pytest.approx(
            {
                "gold_count": len(sample.supporting_facts),
                "match_count": len(relevance_matches),
            }
        )
