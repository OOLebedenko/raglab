from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from raglab.application.evaluation.report import ReportStatus

REPORT_VERSION: Literal[1] = 1


class EvaluationReportDocument(BaseModel):
    """Define the persisted evaluation report schema."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
    )

    report_version: Literal[1] = REPORT_VERSION
    configuration_hash: str
    queries_total: int = Field(gt=0)
    metric_names: list[str] = Field(min_length=1)
    status: ReportStatus
    started_at: str = Field(min_length=1)
    completed_at: str | None = None
    interrupted_at: str | None = None
    error: dict[str, str | int | None] | None = None
    results: list[dict[str, Any]]


@dataclass(frozen=True)
class Chunk:
    """Represent a corpus chunk used by retrieval."""

    text: str
    source: str
    embedding: tuple[float, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
