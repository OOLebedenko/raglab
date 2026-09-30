import pytest

from raglab.application.evaluation.metrics import (
    mean_reciprocal_rank,
    recall_at_k,
    reciprocal_rank,
)
from raglab.application.models import QueryRelevance


def test_recall_at_k() -> None:
    # chunk 0 matches fact 0, chunk 2 matches fact 1;
    # there are 3 gold facts in total.
    query_index = QueryRelevance(
        matches=frozenset(
            {
                (0, 0),
                (2, 1),
            }
        ),
        gold_count=3,
    )

    assert recall_at_k(query_index, k=1) == pytest.approx(1 / 3)
    assert recall_at_k(query_index, k=2) == pytest.approx(1 / 3)
    assert recall_at_k(query_index, k=3) == pytest.approx(2 / 3)


def test_reciprocal_rank() -> None:
    # The first relevant chunk is at zero-based index 2, i.e. rank 3.
    query_index = QueryRelevance(
        matches=frozenset(
            {
                (2, 0),
                (4, 1),
            }
        ),
        gold_count=2,
    )

    assert reciprocal_rank(query_index) == pytest.approx(1 / 3)


def test_mean_reciprocal_rank() -> None:
    reciprocal_ranks = [
        1.0,
        0.5,
        0.25,
    ]

    result = mean_reciprocal_rank(reciprocal_ranks)

    assert result == pytest.approx(sum(reciprocal_ranks) / len(reciprocal_ranks))
