import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

from raglab.application.models import (
    EvaluationRunSummary,
    EvaluationSuccess,
    JudgeFailure,
)

FinalReportStatus = Literal[
    "completed",
    "completed_with_judge_failures",
    "failed",
]

ReportStatus = (
    Literal[
        "in_progress",
        "interrupted",
    ]
    | FinalReportStatus
)

ReportResult = EvaluationSuccess | JudgeFailure


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class EvaluationReport:
    """Track evaluation progress independently of the storage format."""

    configuration_hash: str
    queries_total: int
    metric_names: tuple[str, ...]
    results: dict[int, ReportResult] = field(default_factory=dict)
    status: ReportStatus = "in_progress"
    started_at: str = field(default_factory=_timestamp)
    completed_at: str | None = None
    interrupted_at: str | None = None
    error: dict[str, str | int | None] | None = None

    def __post_init__(self) -> None:
        if self.queries_total < 1:
            raise ValueError("Cannot evaluate an empty benchmark")

        if not self.metric_names or len(set(self.metric_names)) != len(
            self.metric_names
        ):
            raise ValueError("Metric names must be non-empty and unique")

        for index, result in self.results.items():
            self._validate_result(index, result)

    def start(self) -> None:
        """Mark an initial or resumed run as active."""

        self.status = "in_progress"
        self.error = None
        self.interrupted_at = None
        self.completed_at = None

    def pending_query_indices(
        self,
        retry_failed: bool = False,
    ) -> list[int]:
        """Return unprocessed query indices, optionally including judge failures."""

        return [
            index
            for index in range(self.queries_total)
            if index not in self.results
            or (retry_failed and isinstance(self.results[index], JudgeFailure))
        ]

    def record(
        self,
        query_index: int,
        result: ReportResult,
    ) -> None:
        """Record or replace the result for one benchmark sample."""

        self._validate_result(query_index, result)
        self.results[query_index] = result

    def interrupt(
        self,
        query_index: int | None,
        error: BaseException,
    ) -> None:
        """Capture the failure that interrupted the whole experiment."""

        self.status = "interrupted"
        self.interrupted_at = _timestamp()
        self.error = {
            "query_index": query_index,
            "type": type(error).__name__,
            "message": str(error),
        }

    def finalize(self) -> EvaluationRunSummary:
        """Compute final metrics from successful saved results."""

        if len(self.results) != self.queries_total:
            raise ValueError("Cannot finalize an incomplete evaluation")

        successful = [
            result
            for result in self.results.values()
            if isinstance(result, EvaluationSuccess)
        ]

        evaluated = len(successful)
        failed = self.queries_total - evaluated

        status: FinalReportStatus

        if not evaluated:
            status = "failed"
        elif failed:
            status = "completed_with_judge_failures"
        else:
            status = "completed"

        summary = EvaluationRunSummary(
            queries_total=self.queries_total,
            queries_evaluated=evaluated,
            queries_failed=failed,
            metrics={
                name: (
                    sum(result.metrics[name] for result in successful) / evaluated
                    if evaluated
                    else None
                )
                for name in self.metric_names
            },
            status=status,
        )

        self._mark_completed(status)

        return summary

    def _mark_completed(
        self,
        status: FinalReportStatus,
    ) -> None:
        """Mark the experiment as completed."""

        self.status = status
        self.completed_at = _timestamp()
        self.interrupted_at = None
        self.error = None

    def _validate_result(
        self,
        query_index: int,
        result: ReportResult,
    ) -> None:
        if type(query_index) is not int or not 0 <= query_index < self.queries_total:
            raise ValueError(f"Invalid query index: {query_index}")

        if isinstance(result, EvaluationSuccess):
            if set(result.metrics) != set(self.metric_names):
                raise ValueError("Result metrics do not match report configuration")

            if any(not math.isfinite(score) for score in result.metrics.values()):
                raise ValueError("Result metrics must be finite")

        elif not isinstance(result, JudgeFailure):
            raise TypeError(f"Unsupported report result: {type(result).__name__}")
