from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RetrievedChunk:
    """Represent a chunk returned by retrieval."""

    text: str
    source: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RagResult:
    """Represent the result of a RAG query."""

    query: str
    answer: str
    retrieved_chunks: list[RetrievedChunk]
