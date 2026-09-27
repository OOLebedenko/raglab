from collections.abc import Sequence
from pathlib import Path

from bm25s import BM25
from bm25s.tokenization import Tokenizer
from langchain_core.documents import Document

from raglab.infrastructure.tokenization.bm25 import build_tokenizer


def build_bm25_index(
    documents: Sequence[Document],
    path: Path,
) -> None:
    """Build and save a BM25S index from documents"""

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

    # Tokenize the corpus and build the index
    tokenizer = build_tokenizer()

    tokens = tokenizer.tokenize(
        [document.page_content for document in documents],
        return_as="tuple",
        show_progress=False,
    )

    index = BM25(corpus=corpus)
    index.index(tokens, show_progress=False)

    # Save the index, corpus and tokenizer data
    index.save(
        str(path),
        corpus=corpus,
        show_progress=False,
    )

    tokenizer.save_vocab(save_dir=str(path))
    tokenizer.save_stopwords(save_dir=str(path))


def load_bm25_index(path: Path) -> tuple[BM25, Tokenizer]:
    """Load an existing BM25S index and its tokenizer"""

    if not path.is_dir():
        raise FileNotFoundError(f"BM25S index directory not found: {path}")

    index = BM25.load(
        str(path),
        load_corpus=True,
    )

    tokenizer = build_tokenizer()
    tokenizer.load_vocab(str(path))
    tokenizer.load_stopwords(str(path))

    return index, tokenizer
