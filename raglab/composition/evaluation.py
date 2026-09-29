from dataclasses import dataclass
from pathlib import Path

from raglab.application.benchmark_evaluation_service import BenchmarkEvaluationService
from raglab.application.evaluation.report import EvaluationReport
from raglab.application.models import EvaluationSample
from raglab.application.sample_evaluation_service import SampleEvaluationService
from raglab.composition.config import ExperimentConfig, load_config, resolve_path
from raglab.composition.judges import build_judge
from raglab.composition.metrics import build_metrics
from raglab.composition.retrievers import build_retriever
from raglab.infrastructure.data.loaders import load_benchmark
from raglab.infrastructure.data.report import (
    JsonEvaluationReportStore,
    configuration_hash,
)


@dataclass(frozen=True)
class EvaluationExperiment:
    """Group the components required to run an evaluation experiment."""

    service: BenchmarkEvaluationService
    samples: list[EvaluationSample]
    report: EvaluationReport


def _prepare_report(
    config: ExperimentConfig,
    config_path: Path,
    root: Path,
    queries_total: int,
    metric_names: tuple[str, ...],
) -> tuple[JsonEvaluationReportStore, EvaluationReport]:
    """Prepare report storage and restore or create a checkpoint."""

    report_path = (
        root / "artifacts" / "experiments" / f"{config_path.stem}_evaluation.json"
    )

    report_store = JsonEvaluationReportStore(report_path)

    config_hash = configuration_hash(
        config_path=resolve_path(config_path, root),
        benchmark_path=config.data.benchmark,
        chunks_path=config.data.chunks,
    )

    if report_store.exists():
        report = report_store.load(
            configuration_hash=config_hash,
            queries_total=queries_total,
            metric_names=metric_names,
        )
    else:
        report = report_store.create(
            configuration_hash=config_hash,
            queries_total=queries_total,
            metric_names=metric_names,
        )

    return report_store, report


def build_evaluation_experiment(
    config_path: Path,
    project_root: Path,
) -> EvaluationExperiment:
    """Assemble an evaluation experiment with checkpoint support."""

    # 1. Load experiment configuration and benchmark samples
    root = project_root.resolve()

    config = load_config(
        config_path,
        project_root=root,
    )
    samples = load_benchmark(config.data.benchmark)

    if not samples:
        raise ValueError("Cannot evaluate an empty benchmark")

    # 2. Build the configured evaluation metrics
    metrics = build_metrics(config.evaluation.metrics)

    # 3. Prepare report storage and restore or create a checkpoint
    report_store, report = _prepare_report(
        config=config,
        config_path=config_path,
        root=root,
        queries_total=len(samples),
        metric_names=tuple(metric.name for metric in metrics),
    )

    # 4. Assemble the service for evaluating individual queries
    sample_service = SampleEvaluationService(
        retriever=build_retriever(
            config.retriever,
            config.retrieval_policy,
        ),
        relevance_judge=build_judge(config.evaluation.judge),
        metrics=metrics,
    )

    # 5. Assemble the benchmark service and return the experiment
    return EvaluationExperiment(
        service=BenchmarkEvaluationService(
            sample_service=sample_service,
            report_store=report_store,
        ),
        samples=samples,
        report=report,
    )
