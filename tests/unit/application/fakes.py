from raglab.application.models import RetrievedChunk, SupportingFact


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
