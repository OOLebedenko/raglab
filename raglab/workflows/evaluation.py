from pathlib import Path

from raglab.application.models import EvaluationRunSummary
from raglab.composition.evaluation import build_evaluation_experiment


def run_evaluation(
    config_path: Path,
    project_root: Path,
    retry_failed: bool = False,
) -> EvaluationRunSummary:
    """Run retrieval evaluation with checkpoint and resume support."""

    experiment = build_evaluation_experiment(
        config_path=config_path,
        project_root=project_root,
    )

    return experiment.service.run(
        samples=experiment.samples,
        report=experiment.report,
        retry_failed=retry_failed,
    )
