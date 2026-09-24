from raglab.application.models import RagResult
from raglab.application.ports.generator import Generator
from raglab.application.ports.retriever import Retriever


class RagService:
    """Coordinate retrieval and generation for a user query."""

    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
    ) -> None:
        self._retriever = retriever
        self._generator = generator

    def ask(
        self,
        query: str,
    ) -> RagResult:
        retrieved_chunks = self._retriever.retrieve(query)

        answer = self._generator.generate(
            query=query,
            context=retrieved_chunks,
        )

        return RagResult(
            query=query,
            answer=answer,
            retrieved_chunks=retrieved_chunks,
        )
