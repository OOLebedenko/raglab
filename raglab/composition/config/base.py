from pydantic import BaseModel, ConfigDict


class ConfigModel(BaseModel):
    """Base model for immutable configuration."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )
