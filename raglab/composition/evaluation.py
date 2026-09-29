from raglab.application.benchmark_evaluation_service import BenchmarkEvaluationService
from raglab.application.ports.report_store import ReportStore
from raglab.application.sample_evaluation_service import SampleEvaluationService
from raglab.composition.config import ExperimentConfig
from raglab.composition.judges import build_judge
from raglab.composition.metrics import build_metrics
from raglab.composition.retrievers import build_retriever


def build_sample_evaluation_service(
    config: ExperimentConfig,
) -> SampleEvaluationService:
    """Build the service for evaluating a single benchmark sample."""

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


def build_benchmark_evaluation_service(
    config: ExperimentConfig,
    report_store: ReportStore,
) -> BenchmarkEvaluationService:
    """Build the service for running a benchmark with checkpoints."""

    return BenchmarkEvaluationService(
        sample_service=build_sample_evaluation_service(config),
        report_store=report_store,
    )
