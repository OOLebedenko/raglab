from typing import cast
from unittest.mock import Mock, patch

import pytest
from langchain_core.language_models import BaseChatModel

from raglab.composition.config import GeneratorConfig, ModelConfig
from raglab.composition.generators import build_generator
from raglab.infrastructure.generation.prompt import DEFAULT_RAG_PROMPT


@pytest.fixture
def generator_config() -> GeneratorConfig:
    """Provide a generator configuration."""

    return GeneratorConfig(
        model=ModelConfig(
            provider="gigachat",
            name="selected-model",
        ),
    )


@pytest.fixture
def fake_chat_model() -> BaseChatModel:
    """Provide a mock chat model."""

    return cast(BaseChatModel, Mock(spec=BaseChatModel))


def test_build_generator_from_config(
    generator_config: GeneratorConfig,
    fake_chat_model: BaseChatModel,
) -> None:
    """Build the generator model from configuration."""

    with (
        patch(
            "raglab.composition.generators.build_chat_model",
            return_value=fake_chat_model,
        ) as mock_build_model,
        patch(
            "raglab.composition.generators.LangChainGeneratorAdapter"
        ) as mock_adapter,
    ):
        generator = build_generator(generator_config)

    mock_build_model.assert_called_once_with(generator_config.model)
    mock_adapter.assert_called_once_with(
        model=fake_chat_model,
        prompt=DEFAULT_RAG_PROMPT,
    )
    assert generator is mock_adapter.return_value


def test_build_generator_with_injected_model(
    generator_config: GeneratorConfig,
    fake_chat_model: BaseChatModel,
) -> None:
    """Reuse a provided model without creating another one."""

    with (
        patch("raglab.composition.generators.build_chat_model") as mock_build_model,
        patch(
            "raglab.composition.generators.LangChainGeneratorAdapter"
        ) as mock_adapter,
    ):
        build_generator(generator_config, model=fake_chat_model)

    mock_build_model.assert_not_called()
    mock_adapter.assert_called_once_with(
        model=fake_chat_model,
        prompt=DEFAULT_RAG_PROMPT,
    )


def test_build_generator_rejects_unknown_prompt(
    generator_config: GeneratorConfig,
) -> None:
    """Reject an unknown prompt before creating a model."""

    config = generator_config.model_copy(update={"prompt": "unknown_prompt"})

    with patch("raglab.composition.generators.build_chat_model") as mock_build_model:
        with pytest.raises(ValueError, match="Unknown generation prompt"):
            build_generator(config)

    mock_build_model.assert_not_called()
