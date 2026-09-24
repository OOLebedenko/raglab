from raglab.application.models import RetrievedChunk
from raglab.application.rag_service import RagService
from tests.unit.application.fakes import FakeGenerator, FakeRetriever


def test_ask_returns_rag_result(
    retrieved_chunks: list[RetrievedChunk],
) -> None:
    query = "Example query"
    answer = "Generated answer."

    retriever = FakeRetriever(retrieved_chunks)
    generator = FakeGenerator(answer)

    service = RagService(
        retriever=retriever,
        generator=generator,
    )

    result = service.ask(query)

    assert result.query == query
    assert result.answer == answer
    assert result.retrieved_chunks == retrieved_chunks
