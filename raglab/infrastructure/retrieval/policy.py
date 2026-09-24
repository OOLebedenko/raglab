from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalPolicy:
    """Define retrieval behavior."""

    top_k: int
