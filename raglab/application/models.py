from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RetrievedChunk:
    """Represent a chunk returned by retrieval."""

    text: str
    source: str
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SupportingFact:
    """Represent a gold supporting fact for evaluation."""

    text: str
    source: str


@dataclass(frozen=True)
class EvaluationSample:
    """Represent a benchmark sample."""

    query: str
    supporting_facts: list[SupportingFact]


@dataclass(frozen=True)
class QueryRelevance:
    """Represent relevance matches for one query."""

    matches: frozenset[tuple[int, int]]
    gold_count: int


@dataclass(frozen=True)
class EvaluationSuccess:
    """Represent a successfully evaluated benchmark sample."""

    sample: EvaluationSample
    retrieved_chunks: list[RetrievedChunk]
    relevance: QueryRelevance
    metrics: dict[str, float]


@dataclass(frozen=True)
class JudgeFailure:
    """Preserve retrieval results after an expected judge failure."""

    sample: EvaluationSample
    retrieved_chunks: list[RetrievedChunk]
    error_type: str
    error: str


@dataclass(frozen=True)
class RagResult:
    """Represent the result of a RAG query."""

    query: str
    answer: str
    retrieved_chunks: list[RetrievedChunk]
