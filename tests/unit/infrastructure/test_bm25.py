from pathlib import Path

import pytest
from langchain_core.documents import Document

from raglab.infrastructure.indexes.bm25 import (
    build_bm25_index,
    load_bm25_index,
)
from raglab.infrastructure.retrieval.lexical import BM25Retriever


@pytest.mark.parametrize(
    ("k", "expected_count"),
    [
        (1, 1),
        (100, 3),
    ],
)
def test_bm25_save_load_and_retrieve(
    tmp_path: Path,
    k: int,
    expected_count: int,
) -> None:
    documents = [
        Document(
            page_content="Quantum spin resonance in magnetic fields",
            metadata={"source": "physics.md"},
        ),
        Document(
            page_content="Database transactions and isolation levels",
            metadata={"source": "database.md"},
        ),
        Document(
            page_content="Web application architecture",
            metadata={"source": "web.md"},
        ),
    ]

    # Build and save the index
    index_path = tmp_path / "bm25"
    build_bm25_index(documents, index_path)

    # Load the saved index into a new retriever
    index, tokenizer = load_bm25_index(index_path)
    retriever = BM25Retriever(index=index, tokenizer=tokenizer, k=k)

    # Search and verify the returned documents and metadata
    results = retriever.invoke("quantum resonance")

    assert results[0] == documents[0]
    assert len(results) == expected_count


def test_build_bm25_index_rejects_empty_documents(
    tmp_path: Path,
) -> None:
    index_path = tmp_path / "bm25"

    with pytest.raises(
        ValueError,
        match="without documents",
    ):
        build_bm25_index([], index_path)

    assert not index_path.exists()


def test_build_bm25_index_rejects_existing_path(
    tmp_path: Path,
) -> None:
    index_path = tmp_path / "bm25"
    index_path.mkdir()

    documents = [Document(page_content="Example")]

    with pytest.raises(
        FileExistsError,
        match="Index path already exists",
    ):
        build_bm25_index(documents, index_path)


def test_load_bm25_index_requires_existing_directory(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        FileNotFoundError,
        match="BM25S index directory not found",
    ):
        load_bm25_index(tmp_path / "missing")
