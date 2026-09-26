from collections.abc import Sequence
from typing import Self, cast

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, PositiveInt
from rank_bm25 import BM25Okapi


class BM25Retriever(BaseRetriever):  # type: ignore[misc]
    """Retrieve documents using BM25."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    documents: list[Document]
    vectorizer: BM25Okapi
    k: PositiveInt = 4

    @classmethod
    def from_documents(
        cls,
        documents: Sequence[Document],
        *,
        k: int = 4,
    ) -> Self:
        """Build a BM25 index from documents."""

        docs = list(documents)

        if not docs:
            raise ValueError("Cannot build a lexical retriever without documents")

        corpus = [document.page_content.lower().split() for document in docs]

        return cls(
            documents=docs,
            vectorizer=BM25Okapi(corpus),
            k=k,
        )

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """Return the top-k documents for a query."""

        return cast(
            list[Document],
            self.vectorizer.get_top_n(
                query.lower().split(),
                self.documents,
                n=self.k,
            ),
        )
