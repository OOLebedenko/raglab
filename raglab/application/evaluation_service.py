from raglab.application.evaluation.index import build_index
from raglab.application.evaluation.metrics import Metric
from raglab.application.models import EvaluationSample
from raglab.application.ports.relevance import RelevanceJudge
from raglab.application.ports.retriever import Retriever


class EvaluationService:
    """Evaluate retrieval quality on a benchmark."""

    def __init__(
        self,
        retriever: Retriever,
        relevance_judge: RelevanceJudge,
    ) -> None:
        self._retriever = retriever
        self._relevance_judge = relevance_judge

    def evaluate(
        self,
        samples: list[EvaluationSample],
        metrics: list[Metric],
    ) -> dict[str, float]:
        """Evaluate retrieval metrics across benchmark samples."""

        if not samples:
            raise ValueError("Evaluation samples must not be empty.")

        scores: dict[str, list[float]] = {metric.name: [] for metric in metrics}

        for sample in samples:
            retrieved_chunks = self._retriever.retrieve(sample.query)

            query_index = build_index(
                retrieved=retrieved_chunks,
                gold=sample.supporting_facts,
                judge=self._relevance_judge,
            )

            for metric in metrics:
                scores[metric.name].append(metric.calculate(query_index))

        return {name: sum(values) / len(values) for name, values in scores.items()}
