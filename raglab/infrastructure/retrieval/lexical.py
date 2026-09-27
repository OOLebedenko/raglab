from bm25s import BM25
from bm25s.tokenization import Tokenizer
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, PositiveInt


class BM25Retriever(BaseRetriever):  # type: ignore[misc]
    """Retrieve documents from a prepared BM25S index"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    index: BM25
    tokenizer: Tokenizer
    k: PositiveInt = 4

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """Return the top-k documents from the loaded index"""

        # Tokenize the query without modifying the saved vocabulary
        tokens = self.tokenizer.tokenize(
            [query],
            update_vocab=False,
            show_progress=False,
        )

        results, _scores = self.index.retrieve(
            tokens,
            k=min(self.k, len(self.index.corpus)),
            show_progress=False,
        )

        # One query was submitted, so take its batch of results
        return [
            Document(
                page_content=item["text"],
                metadata=item["metadata"],
            )
            for item in results[0]
        ]
