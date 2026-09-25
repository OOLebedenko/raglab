from raglab.composition.config.base import ConfigModel


class ModelConfig(ConfigModel):
    """Configure a chat model."""

    provider: str
    name: str


class GeneratorConfig(ConfigModel):
    """Configure answer generation."""

    model: ModelConfig
    prompt: str = "default_rag"
