import json

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from raglab.application.models import RetrievedChunk, SupportingFact
from raglab.infrastructure.relevance.prompt import RelevancePrompt


class RelevanceResult(BaseModel):
    """Represent relevance matches returned by an LLM."""

    matches: list[tuple[int, int]] = Field(
        description=(
            "Pairs of (chunk_index, fact_index) identifying "
            "retrieved chunks that support the given facts."
        ),
    )


class LangChainRelevanceJudge:
    """Evaluate chunk relevance using a LangChain chat model."""

    def __init__(
        self,
        model: BaseChatModel,
        prompt: RelevancePrompt,
    ) -> None:
        self._model = model
        self._prompt = prompt
        self._parser = PydanticOutputParser(
            pydantic_object=RelevanceResult,
        )

    def find_matches(
        self,
        retrieved: list[RetrievedChunk],
        gold: list[SupportingFact],
    ) -> frozenset[tuple[int, int]]:
        """Find chunk and supporting fact matches using an LLM."""

        if not retrieved or not gold:
            return frozenset()

        retrieved_data = [
            {
                "chunk_index": index,
                "text": chunk.text,
            }
            for index, chunk in enumerate(retrieved)
        ]

        gold_data = [
            {
                "fact_index": index,
                "text": fact.text,
            }
            for index, fact in enumerate(gold)
        ]

        user_prompt = self._prompt.render_user_prompt(
            retrieved=json.dumps(
                retrieved_data,
                ensure_ascii=False,
            ),
            gold=json.dumps(
                gold_data,
                ensure_ascii=False,
            ),
            format_instructions=self._parser.get_format_instructions(),
        )

        messages = [
            SystemMessage(content=self._prompt.system),
            HumanMessage(content=user_prompt),
        ]

        response = self._model.invoke(messages)
        content = response.content

        if not isinstance(content, str):
            raise ValueError("LangChain chat model must return string content.")

        result: RelevanceResult = self._parser.parse(content)

        matches: set[tuple[int, int]] = set()

        for chunk_index, fact_index in result.matches:
            if not (0 <= chunk_index < len(retrieved) and 0 <= fact_index < len(gold)):
                raise ValueError(
                    "LLM returned an invalid relevance index: "
                    f"({chunk_index}, {fact_index})."
                )

            matches.add((chunk_index, fact_index))

        return frozenset(matches)
