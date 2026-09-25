import pytest
from langchain_core.messages import AIMessage

from raglab.application.models import RetrievedChunk, SupportingFact
from raglab.infrastructure.relevance.llm import LangChainRelevanceJudge
from raglab.infrastructure.relevance.prompt import DEFAULT_RELEVANCE_PROMPT


class FakeChatModel:
    """Return a predefined chat model response."""

    def __init__(self, response: str) -> None:
        self._response = response

    def invoke(self, input: object) -> AIMessage:
        return AIMessage(content=self._response)


def test_find_matches_parses_markdown_json(
    retrieved_chunks: list[RetrievedChunk],
    gold: list[SupportingFact],
) -> None:
    model = FakeChatModel(
        response='```json\n{"matches": [[0, 0]]}\n```',
    )
    judge = LangChainRelevanceJudge(
        model=model,
        prompt=DEFAULT_RELEVANCE_PROMPT,
    )

    result = judge.find_matches(retrieved_chunks, gold)

    assert result == frozenset({(0, 0)})


def test_find_matches_raises_for_invalid_json(
    retrieved_chunks: list[RetrievedChunk],
    gold: list[SupportingFact],
) -> None:
    model = FakeChatModel(response="not json")
    judge = LangChainRelevanceJudge(
        model=model,
        prompt=DEFAULT_RELEVANCE_PROMPT,
    )

    with pytest.raises(ValueError, match="Invalid json output"):
        judge.find_matches(retrieved_chunks, gold)
