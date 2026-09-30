from typing import Protocol

from raglab.application.models import RetrievedChunk


class Generator(Protocol):
    """Generate an answer from a query and retrieved context."""

    def generate(
        self,
        query: str,
        context: list[RetrievedChunk],
    ) -> str: ...
