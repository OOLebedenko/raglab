from pathlib import Path

from pydantic import TypeAdapter

from raglab.application.models import EvaluationSample, SupportingFact
from raglab.infrastructure.data.models import Chunk
from raglab.infrastructure.data.validators import (
    BenchmarkSampleRecord,
    ChunkRecord,
)

chunks_validator = TypeAdapter(list[ChunkRecord])
benchmark_validator = TypeAdapter(list[BenchmarkSampleRecord])


def load_chunks(
    path: Path,
) -> list[Chunk]:
    """Load corpus chunks from the RAGLab JSON format.

    Expected format:

    [
        {
            "text": "Chunk text.",
            "source": "document.txt",
            "embedding": [0.1, 0.2, 0.3],
            "metadata": {
                "page": 1
            }
        }
    ]

    `embedding` and `metadata` are optional.
    """

    records = chunks_validator.validate_json(path.read_text(encoding="utf-8"))

    return [
        Chunk(
            text=record.text,
            source=record.source,
            embedding=(
                tuple(record.embedding) if record.embedding is not None else None
            ),
            metadata=record.metadata,
        )
        for record in records
    ]


def load_benchmark(
    path: Path,
) -> list[EvaluationSample]:
    """Load benchmark samples from the RAGLab JSON format.

    Expected format:

    [
        {
            "query": "Example query.",
            "supporting_facts": [
                {
                    "text": "Supporting fact.",
                    "source": "document.txt"
                }
            ]
        }
    ]
    """

    records = benchmark_validator.validate_json(path.read_text(encoding="utf-8"))

    return [
        EvaluationSample(
            query=record.query,
            supporting_facts=[
                SupportingFact(
                    text=fact.text,
                    source=fact.source,
                )
                for fact in record.supporting_facts
            ],
        )
        for record in records
    ]
