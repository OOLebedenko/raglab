from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Chunk:
    """Represent a corpus chunk used by retrieval."""

    text: str
    source: str
    embedding: tuple[float, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
