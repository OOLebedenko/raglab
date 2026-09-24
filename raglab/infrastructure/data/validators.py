from typing import Any

from pydantic import BaseModel, Field


class ChunkRecord(BaseModel):
    """Represent a chunk record in the RAGLab JSON format."""

    text: str
    source: str
    embedding: list[float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SupportingFactRecord(BaseModel):
    """Represent a supporting fact record in the RAGLab JSON format."""

    text: str
    source: str


class BenchmarkSampleRecord(BaseModel):
    """Represent a benchmark sample record in the RAGLab JSON format."""

    query: str
    supporting_facts: list[SupportingFactRecord]
