from typing import cast
from unittest.mock import Mock, patch

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
    """Use a provided model without creating another one"""

    with patch("raglab.composition.judges.build_chat_model") as mock_build_model:
        judge = build_judge(llm_judge_config, model=fake_chat_model)

    assert isinstance(judge, LangChainRelevanceJudge)
    mock_build_model.assert_not_called()


def test_build_llm_judge_from_config(
    llm_judge_config: LlmJudgeConfig,
    fake_chat_model: BaseChatModel,
) -> None:
    """Build the judge model from its configuration"""

    with patch(
        "raglab.composition.judges.build_chat_model",
        return_value=fake_chat_model,
    ) as mock_build_model:
        judge = build_judge(llm_judge_config)

    mock_build_model.assert_called_once_with(llm_judge_config.model)
    assert isinstance(judge, LangChainRelevanceJudge)


def test_build_llm_judge_rejects_unknown_prompt(
    llm_judge_config: LlmJudgeConfig,
    fake_chat_model: BaseChatModel,
) -> None:
    """Reject an unknown relevance prompt."""

    config = llm_judge_config.model_copy(update={"prompt": "custom_prompt"})

    with pytest.raises(ValueError, match="Unknown relevance prompt"):
        build_judge(config, model=fake_chat_model)
