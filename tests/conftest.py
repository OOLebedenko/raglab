import pytest

from raglab.application.models import RetrievedChunk, SupportingFact


@pytest.fixture
def retrieved_chunks() -> list[RetrievedChunk]:
    """Return retrieved chunks for application tests."""

    return [
        RetrievedChunk(
            text="First chunk.",
            source="document.txt",
            score=0.9,
        ),
        RetrievedChunk(
            text="Second chunk.",
            source="another_document.txt",
            score=0.8,
        ),
    ]


@pytest.fixture
def gold() -> list[SupportingFact]:
    """Return gold supporting facts for application tests."""

    return [
        SupportingFact(
            text="First fact.",
            source="document.txt",
        ),
        SupportingFact(
            text="Second fact.",
            source="document.txt",
        ),
        SupportingFact(
            text="Third fact.",
            source="another_document.txt",
        ),
    ]


@pytest.fixture
def relevance_matches() -> frozenset[tuple[int, int]]:
    """Return relevance matches for application tests (chunk_index, fact_index)"""

    return frozenset(
        {
            (0, 1),
            (1, 2),
        }
    )
