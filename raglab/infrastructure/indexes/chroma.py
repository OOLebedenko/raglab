from collections.abc import Sequence
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore


def build_chroma_index(
    documents: Sequence[Document],
    path: Path,
    embeddings: Embeddings,
    options: dict[str, Any],
) -> None:
    """Build and save a local Chroma index"""

    if not documents:
        raise ValueError("Cannot build a vector index without documents")

    if path.exists():
        raise FileExistsError(f"Index path already exists: {path}")

    collection_name = options.get("collection_name")

    if not isinstance(collection_name, str) or not collection_name.strip():
        raise ValueError("Chroma requires a collection_name")

    path.parent.mkdir(parents=True, exist_ok=True)

    Chroma.from_documents(
        documents=list(documents),
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=str(path),
    )


def load_chroma(
    path: Path,
    embeddings: Embeddings,
    options: dict[str, Any],
) -> VectorStore:
    """Connect to an existing local Chroma collection."""

    if not path.is_dir():
        raise FileNotFoundError(f"Chroma index directory not found: {path}")

    collection_name = options.get("collection_name")

    if not isinstance(collection_name, str) or not collection_name:
        raise ValueError("Chroma requires a collection_name")

    return Chroma(
        collection_name=collection_name,
        persist_directory=str(path),
        embedding_function=embeddings,
        create_collection_if_not_exists=False,
    )
