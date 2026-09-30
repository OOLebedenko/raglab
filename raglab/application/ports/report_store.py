from typing import Protocol

from raglab.application.evaluation.report import EvaluationReport


class ReportStore(Protocol):
    """Persist an evaluation report checkpoint."""

    def save(self, report: EvaluationReport) -> None: ...
