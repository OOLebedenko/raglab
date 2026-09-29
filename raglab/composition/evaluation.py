from raglab.application.sample_evaluation_service import SampleEvaluationService
from raglab.composition.config import ExperimentConfig
from raglab.composition.judges import build_judge
from raglab.composition.metrics import build_metrics
from raglab.composition.retrievers import build_retriever


def build_evaluation_service(
    config: ExperimentConfig,
) -> SampleEvaluationService:
    """Assemble the evaluation service from experiment configuration."""

    retriever = build_retriever(
        config.retriever,
        config.retrieval_policy,
    )

    judge = build_judge(config.evaluation.judge)
    metrics = build_metrics(config.evaluation.metrics)

    return SampleEvaluationService(
        retriever=retriever,
        relevance_judge=judge,
        metrics=metrics,
    )
