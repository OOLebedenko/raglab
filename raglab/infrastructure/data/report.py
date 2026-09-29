import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from raglab.application.evaluation.report import EvaluationReport
from raglab.application.models import EvaluationSuccess, JudgeFailure
from raglab.infrastructure.data.models import (
    REPORT_VERSION,
    EvaluationReportDocument,
)
from raglab.infrastructure.relevance.prompt import DEFAULT_RELEVANCE_PROMPT

FINAL_STATUSES = {
    "completed",
    "completed_with_judge_failures",
    "failed",
}

RESULT_ADAPTER: TypeAdapter[EvaluationSuccess | JudgeFailure] = TypeAdapter(
    EvaluationSuccess | JudgeFailure
)


def configuration_hash(
    config_path: Path,
    benchmark_path: Path,
    chunks_path: Path,
) -> str:
    """Identify the experiment configuration, inputs and relevance prompt."""

    payload = {
        "config": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "benchmark": hashlib.sha256(benchmark_path.read_bytes()).hexdigest(),
        "chunks": hashlib.sha256(chunks_path.read_bytes()).hexdigest(),
        "prompt": asdict(DEFAULT_RELEVANCE_PROMPT),
    }

    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


class JsonEvaluationReportStore:
    """Load and atomically persist evaluation checkpoints as JSON."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def exists(self) -> bool:
        """Check whether a checkpoint already exists."""

        return self._path.exists()

    def create(
        self,
        configuration_hash: str,
        queries_total: int,
        metric_names: tuple[str, ...],
    ) -> EvaluationReport:
        """Create and persist a new evaluation report."""

        if self.exists():
            raise FileExistsError(f"Evaluation report already exists: {self._path}")

        report = EvaluationReport(
            configuration_hash=configuration_hash,
            queries_total=queries_total,
            metric_names=metric_names,
        )

        self.save(report)

        return report

    def load(
        self,
        configuration_hash: str,
        queries_total: int,
        metric_names: tuple[str, ...],
    ) -> EvaluationReport:
        """Load and validate an existing evaluation checkpoint."""

        document = EvaluationReportDocument.model_validate_json(
            self._path.read_text(encoding="utf-8")
        )

        self._validate_compatibility(
            document=document,
            configuration_hash=configuration_hash,
            queries_total=queries_total,
            metric_names=metric_names,
        )

        return self._build_report(document)

    def save(self, report: EvaluationReport) -> None:
        """Serialize and atomically persist an evaluation report."""

        document = self._serialize_report(report)
        self._write_atomic(document)

    @classmethod
    def _build_report(
        cls,
        document: EvaluationReportDocument,
    ) -> EvaluationReport:
        """Restore an evaluation report from a validated document."""

        results = cls._deserialize_results(
            saved_results=document.results,
            queries_total=document.queries_total,
        )

        if document.status in FINAL_STATUSES and len(results) != document.queries_total:
            raise ValueError("Completed evaluation report is missing query results")

        return EvaluationReport(
            configuration_hash=document.configuration_hash,
            queries_total=document.queries_total,
            metric_names=tuple(document.metric_names),
            results=results,
            status=document.status,
            started_at=document.started_at,
            completed_at=document.completed_at,
            interrupted_at=document.interrupted_at,
            error=document.error,
        )

    @staticmethod
    def _deserialize_results(
        saved_results: list[dict[str, Any]],
        queries_total: int,
    ) -> dict[int, EvaluationSuccess | JudgeFailure]:
        """Restore typed query results and validate their indices."""

        results: dict[int, EvaluationSuccess | JudgeFailure] = {}

        for saved in saved_results:
            query_index = saved.get("query_index")

            if (
                type(query_index) is not int
                or not 0 <= query_index < queries_total
                or query_index in results
            ):
                raise ValueError(
                    f"Invalid or duplicate saved query index: {query_index}"
                )

            payload = {
                key: value
                for key, value in saved.items()
                if key not in ("query_index", "status")
            }

            result = RESULT_ADAPTER.validate_python(payload)

            expected_status = (
                "judge_failed" if isinstance(result, JudgeFailure) else "success"
            )

            if saved.get("status") != expected_status:
                raise ValueError(f"Invalid status for saved query {query_index}")

            results[query_index] = result

        return results

    @staticmethod
    def _serialize_report(
        report: EvaluationReport,
    ) -> EvaluationReportDocument:
        """Convert an evaluation report into a validated document."""

        results: list[dict[str, Any]] = []

        for query_index, result in sorted(report.results.items()):
            saved = RESULT_ADAPTER.dump_python(
                result,
                mode="json",
            )

            saved["query_index"] = query_index
            saved["status"] = (
                "judge_failed" if isinstance(result, JudgeFailure) else "success"
            )

            results.append(saved)

        return EvaluationReportDocument(
            report_version=REPORT_VERSION,
            configuration_hash=report.configuration_hash,
            queries_total=report.queries_total,
            metric_names=list(report.metric_names),
            status=report.status,
            started_at=report.started_at,
            completed_at=report.completed_at,
            interrupted_at=report.interrupted_at,
            error=report.error,
            results=results,
        )

    @staticmethod
    def _validate_compatibility(
        document: EvaluationReportDocument,
        configuration_hash: str,
        queries_total: int,
        metric_names: tuple[str, ...],
    ) -> None:
        """Check compatibility with the current experiment."""

        if document.configuration_hash != configuration_hash:
            raise ValueError("Evaluation configuration has changed; cannot resume")

        if document.queries_total != queries_total:
            raise ValueError("Benchmark size has changed; cannot resume")

        if document.metric_names != list(metric_names):
            raise ValueError("Evaluation metrics have changed; cannot resume")

    def _write_atomic(
        self,
        document: EvaluationReportDocument,
    ) -> None:
        """Write JSON using a temporary file and atomic replacement."""

        content = document.model_dump_json(indent=2)

        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self._path.parent,
                prefix=f".{self._path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                temporary.write(content)
                temporary.flush()
                os.fsync(temporary.fileno())

            temporary_path.replace(self._path)

        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
