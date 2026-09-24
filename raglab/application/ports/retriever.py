from typing import Protocol

from raglab.application.models import RetrievedChunk


class Retriever(Protocol):
    """Retrieve chunks relevant to a query."""

    def retrieve(
        self,
        query: str,
    ) -> list[RetrievedChunk]: ...
