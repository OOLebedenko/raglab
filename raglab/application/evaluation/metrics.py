from collections.abc import Callable
from dataclasses import dataclass

from raglab.application.models import QueryRelevanceIndex

MetricFunction = Callable[[QueryRelevanceIndex], float]


@dataclass(frozen=True)
class Metric:
    """Represent a named evaluation metric."""

    name: str
    calculate: MetricFunction


def recall_at_k(
    index: QueryRelevanceIndex,
    k: int,
) -> float:
    """Calculate recall among the top-k retrieved chunks."""

    if index.gold_count == 0:
        return 0.0

    found_facts = {
        fact_index for chunk_index, fact_index in index.matches if chunk_index < k
    }

    return len(found_facts) / index.gold_count


def reciprocal_rank(
    index: QueryRelevanceIndex,
) -> float:
    """Calculate reciprocal rank for one query."""

    if not index.matches:
        return 0.0

    first_chunk_index = min(chunk_index for chunk_index, _ in index.matches)

    return 1.0 / (first_chunk_index + 1)


def mean_reciprocal_rank(
    reciprocal_ranks: list[float],
) -> float:
    """Calculate mean reciprocal rank across queries."""

    if not reciprocal_ranks:
        return 0.0

    return sum(reciprocal_ranks) / len(reciprocal_ranks)
