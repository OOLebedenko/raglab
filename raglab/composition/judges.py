from typing import assert_never

from langchain_core.language_models import BaseChatModel

from raglab.composition.chat_models import build_chat_model
from raglab.composition.config import (
    JudgeConfig,
    LlmJudgeConfig,
    SubstringJudgeConfig,
)
from raglab.infrastructure.relevance.llm import LangChainRelevanceJudge
from raglab.infrastructure.relevance.prompt import DEFAULT_RELEVANCE_PROMPT
from raglab.infrastructure.relevance.substring import SubstringRelevanceJudge


def _validate_llm_judge_prompt(config: LlmJudgeConfig) -> None:
    """Validate LLM judge configuration."""

    if config.prompt != "default_relevance":
        raise ValueError(f"Unknown relevance prompt: {config.prompt}")


def _build_judge(
    config: JudgeConfig,
    model: BaseChatModel | None,
) -> SubstringRelevanceJudge | LangChainRelevanceJudge:
    """Build the selected relevance judge."""

    if isinstance(config, SubstringJudgeConfig):
        return SubstringRelevanceJudge()

    if isinstance(config, LlmJudgeConfig):
        _validate_llm_judge_prompt(config)

        chat_model = model if model is not None else build_chat_model(config.model)

        return LangChainRelevanceJudge(
            model=chat_model,
            prompt=DEFAULT_RELEVANCE_PROMPT,
        )

    assert_never(config)


def build_judge(
    config: JudgeConfig,
    *,
    model: BaseChatModel | None = None,
) -> SubstringRelevanceJudge | LangChainRelevanceJudge:
    """Build a relevance judge from configuration."""

    return _build_judge(config, model)
