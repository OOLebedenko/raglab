import pytest

from raglab.application.models import RetrievedChunk
from raglab.application.rag_service import RagService


class FakeRetriever:
    """Return predefined chunks."""

    def __init__(
        self,
        chunks: list[RetrievedChunk],
    ) -> None:
        self._chunks = chunks

    def retrieve(
        self,
        query: str,
    ) -> list[RetrievedChunk]:
        return self._chunks


class FakeGenerator:
    """Return a predefined answer."""

    def __init__(
        self,
        answer: str,
    ) -> None:
        self._answer = answer

    def generate(
        self,
        query: str,
        context: list[RetrievedChunk],
    ) -> str:
        return self._answer


@pytest.fixture
def chunks() -> list[RetrievedChunk]:
    """Return retrieved chunks for tests."""

    return [
        RetrievedChunk(
            text="Relevant document content.",
            source="document.txt",
            score=0.91,
        ),
        RetrievedChunk(
            text="Another relevant passage.",
            source="another_document.txt",
            score=0.84,
        ),
    ]


def test_ask_returns_rag_result(
    chunks: list[RetrievedChunk],
) -> None:
    query = "Example query"
    answer = "Generated answer."

    service = RagService(
        retriever=FakeRetriever(chunks),
        generator=FakeGenerator(answer),
    )

    result = service.ask(query)

    assert result.query == query
    assert result.answer == answer
    assert result.retrieved_chunks == chunks
