from langchain_core.language_models import BaseChatModel

from raglab.composition.chat_models import build_chat_model
from raglab.composition.config import GeneratorConfig
from raglab.infrastructure.generation.langchain import LangChainGeneratorAdapter
from raglab.infrastructure.generation.prompt import (
    DEFAULT_RAG_PROMPT,
    GenerationPrompt,
)


def _resolve_generator_prompt(config: GeneratorConfig) -> GenerationPrompt:
    """Resolve the configured generation prompt."""

    if config.prompt == "default_rag":
        return DEFAULT_RAG_PROMPT

    raise ValueError(f"Unknown generation prompt: {config.prompt}")


def build_generator(
    config: GeneratorConfig,
    *,
    model: BaseChatModel | None = None,
) -> LangChainGeneratorAdapter:
    """Build a generator from configuration."""

    prompt = _resolve_generator_prompt(config)
    chat_model = model if model is not None else build_chat_model(config.model)

    return LangChainGeneratorAdapter(
        model=chat_model,
        prompt=prompt,
    )
