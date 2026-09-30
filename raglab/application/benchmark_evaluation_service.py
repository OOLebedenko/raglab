import logging

from raglab.application.evaluation.report import EvaluationReport
from raglab.application.models import (
    EvaluationRunSummary,
    EvaluationSample,
    JudgeFailure,
)
from raglab.application.ports.report_store import ReportStore
from raglab.application.sample_evaluation_service import SampleEvaluationService

logger = logging.getLogger(__name__)


class BenchmarkEvaluationService:
    """Run a benchmark and save a checkpoint after every sample."""

    def __init__(
        self,
        sample_service: SampleEvaluationService,
        report_store: ReportStore,
    ) -> None:
        self._sample_service = sample_service
        self._report_store = report_store

    def run(
        self,
        samples: list[EvaluationSample],
        report: EvaluationReport,
        retry_failed: bool = False,
    ) -> EvaluationRunSummary:
        """Evaluate pending samples, then finalize the report."""

        if len(samples) != report.queries_total:
            raise ValueError("Benchmark size does not match the report")

        current_query_index: int | None = None

        try:
            report.start()
            self._report_store.save(report)

            for query_index in report.pending_query_indices(
                retry_failed=retry_failed,
            ):
                current_query_index = query_index

                result = self._sample_service.evaluate_sample(samples[query_index])

                report.record(query_index, result)
                self._report_store.save(report)

                if isinstance(result, JudgeFailure):
                    logger.warning(
                        "Relevance judge failed for query %d/%d: %s",
                        query_index + 1,
                        len(samples),
                        result.error,
                    )

            summary = report.finalize()
            self._report_store.save(report)

            return summary

        except (Exception, KeyboardInterrupt) as exc:
            report.interrupt(current_query_index, exc)
            self._report_store.save(report)
            raise
