from collections.abc import Sequence
from pathlib import Path
from typing import Self

from bm25s import BM25
from bm25s.tokenization import Tokenizer
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, PositiveInt


def _tokenize(text: str) -> list[str]:
    """Tokenize text using lowercasing and whitespace splitting."""

    return text.lower().split()


def build_tokenizer() -> Tokenizer:
    """Create the tokenizer shared by indexing and retrieval."""

    return Tokenizer(
        splitter=_tokenize,
        stopwords=[],
    )


def build_bm25_index(
    documents: Sequence[Document],
    path: Path,
) -> None:
    """Build and save a BM25S index from documents."""

    if not documents:
        raise ValueError("Cannot build a lexical index without documents")

    if path.exists():
        raise FileExistsError(f"Index path already exists: {path}")

    path.mkdir(parents=True)

    corpus = [
        {
            "text": document.page_content,
            "metadata": document.metadata,
        }
        for document in documents
    ]

    # Tokenize the corpus and build the index.
    tokenizer = build_tokenizer()

    tokens = tokenizer.tokenize(
        [document.page_content for document in documents],
        return_as="tuple",
        show_progress=False,
    )

    index = BM25(corpus=corpus)
    index.index(tokens, show_progress=False)

    # Save the index, corpus and tokenizer data.
    index.save(
        str(path),
        corpus=corpus,
        show_progress=False,
    )

    tokenizer.save_vocab(save_dir=str(path))
    tokenizer.save_stopwords(save_dir=str(path))


class BM25Retriever(BaseRetriever):  # type: ignore[misc]
    """Retrieve documents from a prepared BM25S index."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    index: BM25
    tokenizer: Tokenizer
    k: PositiveInt = 4

    @classmethod
    def from_index(
        cls,
        path: Path,
        *,
        k: int = 4,
    ) -> Self:
        """Load an existing BM25S index and its tokenizer."""

        if not path.is_dir():
            raise FileNotFoundError(f"BM25S index directory not found: {path}")

        # Load the prepared index and its corpus.
        index = BM25.load(
            str(path),
            load_corpus=True,
        )

        # Restore the vocabulary and stopwords used during indexing.
        tokenizer = build_tokenizer()
        tokenizer.load_vocab(str(path))
        tokenizer.load_stopwords(str(path))

        return cls(
            index=index,
            tokenizer=tokenizer,
            k=k,
        )

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """Return the top-k documents from the loaded index."""

        # Tokenize the query without modifying the saved vocabulary.
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

        # One query was submitted, so take its batch of results.
        return [
            Document(
                page_content=item["text"],
                metadata=item["metadata"],
            )
            for item in results[0]
        ]
