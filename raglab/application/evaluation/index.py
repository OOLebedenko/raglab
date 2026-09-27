from raglab.application.models import (
    QueryRelevance,
    RetrievedChunk,
    SupportingFact,
)
from raglab.application.ports.relevance import RelevanceJudge


def build_index(
    retrieved: list[RetrievedChunk],
    gold: list[SupportingFact],
    judge: RelevanceJudge,
) -> QueryRelevance:
    """Build relevance index for one query."""

    return QueryRelevance(
        matches=judge.find_matches(
            retrieved=retrieved,
            gold=gold,
        ),
        gold_count=len(gold),
    )
