from typing import assert_never

from langchain_core.language_models import BaseChatModel

from raglab.composition.config import (
    JudgeConfig,
    LlmJudgeConfig,
    SubstringJudgeConfig,
)
from raglab.infrastructure.relevance.llm import LangChainRelevanceJudge
from raglab.infrastructure.relevance.prompt import DEFAULT_RELEVANCE_PROMPT
from raglab.infrastructure.relevance.substring import SubstringRelevanceJudge


def _validate_llm_judge(
    config: LlmJudgeConfig,
    model: BaseChatModel | None,
) -> BaseChatModel:
    """Validate LLM judge dependencies."""

    if model is None:
        raise ValueError("LLM judge requires a chat model")

    if config.prompt != "default_relevance":
        raise ValueError(f"Unknown relevance prompt: {config.prompt}")

    return model


def _build_judge(
    config: JudgeConfig,
    model: BaseChatModel | None,
) -> SubstringRelevanceJudge | LangChainRelevanceJudge:
    """Build a judge for the selected configuration."""

    if isinstance(config, SubstringJudgeConfig):
        return SubstringRelevanceJudge()

    if isinstance(config, LlmJudgeConfig):
        validated_model = _validate_llm_judge(config, model)

        return LangChainRelevanceJudge(
            model=validated_model,
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
