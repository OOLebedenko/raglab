from collections.abc import Callable

from langchain_core.language_models import BaseChatModel
from langchain_gigachat import GigaChat

from raglab.composition.config import ModelConfig

# TODO: use init_chat_model for standard providers when needed
_MODEL_BUILDERS: dict[str, Callable[[str], BaseChatModel]] = {
    "gigachat": lambda name: GigaChat(model=name),
}


def build_chat_model(config: ModelConfig) -> BaseChatModel:
    """Build a chat model from configuration."""

    try:
        builder = _MODEL_BUILDERS[config.provider]
    except KeyError as exc:
        raise ValueError(f"Unsupported chat model provider: {config.provider}") from exc

    return builder(config.name)
