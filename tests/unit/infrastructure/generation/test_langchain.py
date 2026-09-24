from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage

from raglab.application.models import RetrievedChunk
from raglab.infrastructure.generation.langchain import (
    LangChainGeneratorAdapter,
    format_context,
)
from raglab.infrastructure.generation.prompt import GenerationPrompt


@dataclass
class FakeResponse:
    """Represent a fake chat model response."""

    content: str


class FakeChatModel:
    """Return a predefined response and record model input."""

    def __init__(
        self,
        response: str,
    ) -> None:
        self._response = response
        self.input: object | None = None

    def invoke(
        self,
        input: object,
    ) -> FakeResponse:
        self.input = input

        return FakeResponse(
            content=self._response,
        )


def test_format_context() -> None:
    chunks = [
        RetrievedChunk(
            text="First chunk.",
            source="first.txt",
        ),
        RetrievedChunk(
            text="Second chunk.",
            source="second.txt",
        ),
    ]

    result = format_context(chunks)

    assert result == "First chunk.\n\nSecond chunk."


def test_generate_builds_messages_and_returns_answer() -> None:
    context = [
        RetrievedChunk(
            text="Example context.",
            source="document.txt",
        ),
    ]
    prompt = GenerationPrompt(
        system="Answer using the context.",
        user_template=("Context:\n{context}\n\nQuestion:\n{query}"),
    )
    model = FakeChatModel(
        response="Generated answer.",
    )
    generator = LangChainGeneratorAdapter(
        model=model,
        prompt=prompt,
    )

    result = generator.generate(
        query="Example question?",
        context=context,
    )

    assert result == "Generated answer."
    assert model.input == [
        SystemMessage(
            content="Answer using the context.",
        ),
        HumanMessage(
            content=("Context:\nExample context.\n\nQuestion:\nExample question?"),
        ),
    ]
