import json
import logging

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from raglab.application.errors import RelevanceJudgeError
from raglab.application.models import RetrievedChunk, SupportingFact
from raglab.infrastructure.relevance.prompt import RelevancePrompt

logger = logging.getLogger(__name__)


class JudgeOutputError(RelevanceJudgeError):
    """Indicate that the relevance judge returned an invalid response."""


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
        max_attempts: int = 3,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

        if max_attempts > 1 and prompt.retry_template is None:
            raise ValueError("A retry prompt is required when max_attempts > 1")

        self._model = model
        self._prompt = prompt
        self._max_attempts = max_attempts
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

        messages = self._build_messages(retrieved, gold)

        return self._invoke_with_retry(
            messages=messages,
            chunk_count=len(retrieved),
            fact_count=len(gold),
        )

    def _build_messages(
        self,
        retrieved: list[RetrievedChunk],
        gold: list[SupportingFact],
    ) -> list[BaseMessage]:
        """Prepare the initial relevance evaluation messages."""

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
            chunk_count=len(retrieved),
            fact_count=len(gold),
        )

        return [
            SystemMessage(content=self._prompt.system),
            HumanMessage(content=user_prompt),
        ]

    def _invoke_with_retry(
        self,
        messages: list[BaseMessage],
        chunk_count: int,
        fact_count: int,
    ) -> frozenset[tuple[int, int]]:
        """Retry invalid responses with validation feedback."""

        history = list(messages)
        last_error: JudgeOutputError | None = None

        for attempt in range(1, self._max_attempts + 1):
            response = self._model.invoke(history)

            try:
                return self._parse_matches(
                    content=response.content,
                    chunk_count=chunk_count,
                    fact_count=fact_count,
                )

            except JudgeOutputError as exc:
                last_error = exc

                self._log_invalid_response(
                    attempt=attempt,
                    error=exc,
                    content=response.content,
                )

                if attempt < self._max_attempts:
                    self._append_retry(
                        history=history,
                        response=response,
                        error=exc,
                        chunk_count=chunk_count,
                        fact_count=fact_count,
                    )

        raise JudgeOutputError(
            "Relevance judge failed to produce a valid response "
            f"after {self._max_attempts} attempts. "
            f"Last error: {last_error}"
        ) from last_error

    def _append_retry(
        self,
        history: list[BaseMessage],
        response: BaseMessage,
        error: JudgeOutputError,
        chunk_count: int,
        fact_count: int,
    ) -> None:
        """Append an invalid response and corrective feedback."""

        retry_prompt = self._prompt.render_retry_prompt(
            error=str(error),
            chunk_count=chunk_count,
            fact_count=fact_count,
        )

        history.extend(
            [
                response,
                HumanMessage(content=retry_prompt),
            ]
        )

    def _log_invalid_response(
        self,
        attempt: int,
        error: JudgeOutputError,
        content: object,
    ) -> None:
        """Log an invalid response and its validation error."""

        logger.warning(
            "Invalid relevance judge output: attempt=%d/%d, error=%s",
            attempt,
            self._max_attempts,
            error,
        )

        logger.debug(
            "Invalid relevance judge raw response: %.500r",
            content,
        )

    def _parse_matches(
        self,
        content: object,
        chunk_count: int,
        fact_count: int,
    ) -> frozenset[tuple[int, int]]:
        """Parse and validate a relevance judge response."""

        if not isinstance(content, str):
            raise JudgeOutputError("The relevance judge must return string content")

        try:
            result: RelevanceResult = self._parser.parse(content)

        except OutputParserException as exc:
            raise JudgeOutputError(
                "Invalid JSON output from relevance judge: "
                'expected a JSON object with a "matches" field'
            ) from exc

        matches: set[tuple[int, int]] = set()

        for chunk_index, fact_index in result.matches:
            if not (0 <= chunk_index < chunk_count):
                raise JudgeOutputError(
                    f"Invalid chunk index: {chunk_index}. "
                    f"Valid indices: {list(range(chunk_count))}"
                )

            if not (0 <= fact_index < fact_count):
                raise JudgeOutputError(
                    f"Invalid fact index: {fact_index}. "
                    f"Valid indices: {list(range(fact_count))}"
                )

            matches.add((chunk_index, fact_index))

        return frozenset(matches)
