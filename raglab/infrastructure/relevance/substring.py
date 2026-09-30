from raglab.application.models import RetrievedChunk, SupportingFact


class SubstringRelevanceJudge:
    """Match supporting facts contained in retrieved chunks."""

    def find_matches(
        self,
        retrieved: list[RetrievedChunk],
        gold: list[SupportingFact],
    ) -> frozenset[tuple[int, int]]:
        """Find matching chunk and supporting fact pairs."""

        matches: set[tuple[int, int]] = set()

        for chunk_index, chunk in enumerate(retrieved):
            chunk_text = self._normalize(chunk.text)

            for fact_index, fact in enumerate(gold):
                fact_text = self._normalize(fact.text)

                if fact_text and fact_text in chunk_text:
                    matches.add((chunk_index, fact_index))

        return frozenset(matches)

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize case and whitespace."""

        return " ".join(text.casefold().split())
