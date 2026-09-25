from typing import cast
from unittest.mock import Mock

import pytest
from langchain_core.language_models import BaseChatModel

from raglab.composition.config import (
    LlmJudgeConfig,
    ModelConfig,
    SubstringJudgeConfig,
)
from raglab.composition.judges import build_judge
from raglab.infrastructure.relevance.llm import LangChainRelevanceJudge
from raglab.infrastructure.relevance.substring import SubstringRelevanceJudge


@pytest.fixture
def llm_judge_config() -> LlmJudgeConfig:
    """Provide an LLM judge configuration."""

    return LlmJudgeConfig(
        type="llm",
        model=ModelConfig(
            provider="gigachat",
            name="GigaChat-2-Max",
        ),
    )


@pytest.fixture
def fake_chat_model() -> BaseChatModel:
    """Provide a mock chat model."""

    return cast(BaseChatModel, Mock(spec=BaseChatModel))


def test_build_substring_judge() -> None:
    """Build a substring judge without an LLM."""

    config = SubstringJudgeConfig(type="substring")

    judge = build_judge(config)

    assert isinstance(judge, SubstringRelevanceJudge)


def test_build_llm_judge(
    llm_judge_config: LlmJudgeConfig,
    fake_chat_model: BaseChatModel,
) -> None:
    """Build an LLM judge with the provided model."""

    judge = build_judge(llm_judge_config, model=fake_chat_model)

    assert isinstance(judge, LangChainRelevanceJudge)


def test_build_llm_judge_requires_model(
    llm_judge_config: LlmJudgeConfig,
) -> None:
    """Reject an LLM judge without a chat model."""

    with pytest.raises(ValueError, match="requires a chat model"):
        build_judge(llm_judge_config)


def test_build_llm_judge_rejects_unknown_prompt(
    llm_judge_config: LlmJudgeConfig,
    fake_chat_model: BaseChatModel,
) -> None:
    """Reject an unknown relevance prompt."""

    config = llm_judge_config.model_copy(update={"prompt": "custom_prompt"})

    with pytest.raises(ValueError, match="Unknown relevance prompt"):
        build_judge(config, model=fake_chat_model)
