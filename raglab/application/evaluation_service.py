from raglab.application.errors import RelevanceJudgeError
from raglab.application.evaluation.index import build_index
from raglab.application.evaluation.metrics import Metric
from raglab.application.models import (
    EvaluationSample,
    EvaluationSuccess,
    JudgeFailure,
)
from raglab.application.ports.relevance import RelevanceJudge
from raglab.application.ports.retriever import Retriever


class EvaluationService:
    """Evaluate retrieval quality for benchmark samples."""

    def __init__(
        self,
        retriever: Retriever,
        relevance_judge: RelevanceJudge,
        metrics: list[Metric] | None = None,
    ) -> None:
        self._retriever = retriever
        self._relevance_judge = relevance_judge
        self._metrics = metrics

    def evaluate_sample(
        self,
        sample: EvaluationSample,
        metrics: list[Metric] | None = None,
    ) -> EvaluationSuccess | JudgeFailure:
        """Retrieve, judge and score one benchmark sample."""

        selected_metrics = self._metrics if metrics is None else metrics

        if selected_metrics is None:
            raise ValueError("Evaluation metrics must be provided")

        retrieved = self._retriever.retrieve(sample.query)

        try:
            relevance = build_index(
                retrieved=retrieved,
                gold=sample.supporting_facts,
                judge=self._relevance_judge,
            )
        except RelevanceJudgeError as exc:
            return JudgeFailure(
                sample=sample,
                retrieved_chunks=retrieved,
                error_type=type(exc).__name__,
                error=str(exc),
            )

        return EvaluationSuccess(
            sample=sample,
            retrieved_chunks=retrieved,
            matches=relevance.matches,
            metrics={
                metric.name: float(metric.calculate(relevance))
                for metric in selected_metrics
            },
        )

    def evaluate(
        self,
        samples: list[EvaluationSample],
        metrics: list[Metric] | None = None,
    ) -> dict[str, float]:
        """Aggregate successful sample metrics for legacy callers."""

        if not samples:
            raise ValueError("Evaluation samples must not be empty")

        selected_metrics = self._metrics if metrics is None else metrics

        if selected_metrics is None:
            raise ValueError("Evaluation metrics must be provided")

        totals = {metric.name: 0.0 for metric in selected_metrics}
        evaluated = 0

        for sample in samples:
            result = self.evaluate_sample(
                sample,
                metrics=selected_metrics,
            )

            if isinstance(result, JudgeFailure):
                continue

            for name, value in result.metrics.items():
                totals[name] += value

            evaluated += 1

        if not evaluated:
            raise ValueError("No queries were successfully evaluated")

        return {name: total / evaluated for name, total in totals.items()}
