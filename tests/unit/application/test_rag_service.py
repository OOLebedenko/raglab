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


def test_ask_returns_rag_result(
    retrieved_chunks: list[RetrievedChunk],
) -> None:
    query = "Example query"
    answer = "Generated answer."

    service = RagService(
        retriever=FakeRetriever(retrieved_chunks),
        generator=FakeGenerator(answer),
    )

    result = service.ask(query)

    assert result.query == query
    assert result.answer == answer
    assert result.retrieved_chunks == retrieved_chunks
