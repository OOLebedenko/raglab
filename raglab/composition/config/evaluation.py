from typing import Literal

from pydantic import PositiveInt

from raglab.composition.config.base import ConfigModel
from raglab.composition.config.generation import ModelConfig


class SubstringJudgeConfig(ConfigModel):
    """Configure substring relevance matching."""

    type: Literal["substring"]


class LlmJudgeConfig(ConfigModel):
    """Configure LLM-based relevance evaluation."""

    type: Literal["llm"]
    model: ModelConfig
    prompt: str = "default_relevance"
    max_attempts: PositiveInt = 3


class RecallAtKConfig(ConfigModel):
    """Configure recall at k."""

    type: Literal["recall_at_k"]
    k: PositiveInt


class ReciprocalRankConfig(ConfigModel):
    """Configure reciprocal rank."""

    type: Literal["reciprocal_rank"]
