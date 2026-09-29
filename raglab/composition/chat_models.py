import os
from collections.abc import Callable

from langchain_core.language_models import BaseChatModel
from langchain_gigachat import GigaChat

from raglab.composition.config import ModelConfig


def _build_gigachat(name: str) -> BaseChatModel:
    return GigaChat(
        credentials=os.environ["GIGACHAT_AUTH_KEY"],
        scope="GIGACHAT_API_PERS",
        base_url="https://api.giga.chat/v1",
        model=name,
        temperature=0,
    )


_MODEL_BUILDERS: dict[str, Callable[[str], BaseChatModel]] = {
    "gigachat": _build_gigachat,
}


def build_chat_model(config: ModelConfig) -> BaseChatModel:
    """Build a chat model from configuration."""

    try:
        builder = _MODEL_BUILDERS[config.provider]
    except KeyError as exc:
        raise ValueError(f"Unsupported chat model provider: {config.provider}") from exc

    return builder(config.name)
