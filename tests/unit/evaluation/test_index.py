from raglab.application.evaluation.index import build_index
from raglab.application.models import RetrievedChunk, SupportingFact


class FakeRelevanceJudge:
    """Return predefined relevance matches."""

    def __init__(
        self,
        matches: frozenset[tuple[int, int]],
    ) -> None:
        self._matches = matches

    def find_matches(
        self,
        retrieved: list[RetrievedChunk],
        gold: list[SupportingFact],
    ) -> frozenset[tuple[int, int]]:
        return self._matches


def test_build_index_returns_query_relevance_index(
    retrieved_chunks: list[RetrievedChunk],
    gold: list[SupportingFact],
    relevance_matches: frozenset[tuple[int, int]],
) -> None:
    result = build_index(
        retrieved=retrieved_chunks,
        gold=gold,
        judge=FakeRelevanceJudge(relevance_matches),
    )

    assert result.matches == relevance_matches
    assert result.gold_count == len(gold)
