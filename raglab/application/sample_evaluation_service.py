from raglab.application.errors import RelevanceJudgeError
from raglab.application.evaluation.metrics import Metric
from raglab.application.models import (
    EvaluationSample,
    EvaluationSuccess,
    JudgeFailure,
    QueryRelevance,
)
from raglab.application.ports.relevance import RelevanceJudge
from raglab.application.ports.retriever import Retriever


class SampleEvaluationService:
    """Retrieve, judge and score one benchmark sample."""

    def __init__(
        self,
        retriever: Retriever,
        relevance_judge: RelevanceJudge,
        metrics: list[Metric],
    ) -> None:
        if not metrics:
            raise ValueError("Evaluation metrics must not be empty")

        self._retriever = retriever
        self._relevance_judge = relevance_judge
        self._metrics = metrics

    def evaluate_sample(
        self,
        sample: EvaluationSample,
    ) -> EvaluationSuccess | JudgeFailure:
        """Return a typed success or an expected judge failure."""

        retrieved = self._retriever.retrieve(sample.query)

        try:
            matches = self._relevance_judge.find_matches(
                retrieved=retrieved,
                gold=sample.supporting_facts,
            )
        except RelevanceJudgeError as exc:
            return JudgeFailure(
                sample=sample,
                retrieved_chunks=retrieved,
                error_type=type(exc).__name__,
                error=str(exc),
            )

        relevance = QueryRelevance(
            matches=matches,
            gold_count=len(sample.supporting_facts),
        )

        return EvaluationSuccess(
            sample=sample,
            retrieved_chunks=retrieved,
            relevance=relevance,
            metrics={
                metric.name: float(metric.calculate(relevance))
                for metric in self._metrics
            },
        )
