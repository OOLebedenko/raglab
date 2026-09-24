from typing import Protocol

from raglab.application.models import RetrievedChunk, SupportingFact


class RelevanceJudge(Protocol):
    """Find relevance matches between retrieved chunks and gold facts."""

    def find_matches(
        self,
        retrieved: list[RetrievedChunk],
        gold: list[SupportingFact],
    ) -> frozenset[tuple[int, int]]: ...
