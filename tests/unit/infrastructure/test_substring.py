from dataclasses import replace

from raglab.application.models import RetrievedChunk, SupportingFact
from raglab.infrastructure.relevance.substring import SubstringRelevanceJudge


def test_find_matches_returns_chunk_fact_pairs(
    retrieved_chunks: list[RetrievedChunk],
    gold: list[SupportingFact],
) -> None:
    retrieved = [
        replace(retrieved_chunks[0], text=gold[0].text),
        replace(
            retrieved_chunks[1],
            text=f"{gold[0].text} {gold[1].text}",
        ),
    ]

    judge = SubstringRelevanceJudge()

    result = judge.find_matches(retrieved, gold)

    expected_matches = frozenset(
        {
            (0, 0),  # First chunk contains First fact
            (1, 0),  # Second chunk contains First fact
            (1, 1),  # Second chunk contains Second fact
        }
    )

    assert result == expected_matches


def test_find_matches_normalizes_case_and_whitespace(
    retrieved_chunks: list[RetrievedChunk],
    gold: list[SupportingFact],
) -> None:
    retrieved = [
        replace(
            retrieved_chunks[0],
            text=" FIRST   \n FACT. ",
        ),
    ]

    judge = SubstringRelevanceJudge()

    result = judge.find_matches(retrieved, gold)

    assert result == frozenset({(0, 0)})


def test_find_matches_returns_empty_for_nonmatching_facts(
    retrieved_chunks: list[RetrievedChunk],
    gold: list[SupportingFact],
) -> None:
    judge = SubstringRelevanceJudge()

    result = judge.find_matches(retrieved_chunks, gold)

    assert result == frozenset()
