import pytest
from langchain_core.messages import AIMessage

from raglab.application.models import RetrievedChunk, SupportingFact
from raglab.infrastructure.relevance.llm import (
    JudgeOutputError,
    LangChainRelevanceJudge,
)
from raglab.infrastructure.relevance.prompt import DEFAULT_RELEVANCE_PROMPT


class FakeChatModel:
    """Return a predefined chat model response."""

    def __init__(self, response: str) -> None:
        self._response = response
        self.calls = 0

    def invoke(self, input: object) -> AIMessage:
        self.calls += 1
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
    assert model.calls == 1


def test_find_matches_raises_for_invalid_json(
    retrieved_chunks: list[RetrievedChunk],
    gold: list[SupportingFact],
) -> None:
    model = FakeChatModel(response="not json")
    judge = LangChainRelevanceJudge(
        model=model,
        prompt=DEFAULT_RELEVANCE_PROMPT,
        max_attempts=3,
    )

    with pytest.raises(
        JudgeOutputError,
        match="failed to produce a valid response after 3 attempts",
    ) as exc_info:
        judge.find_matches(retrieved_chunks, gold)

    assert model.calls == 3
    assert isinstance(exc_info.value.__cause__, JudgeOutputError)
    assert "Invalid JSON output" in str(exc_info.value.__cause__)
