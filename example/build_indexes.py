"""Build BM25S and BGE-M3 indexes for the Cendara experiment."""

import json
from pathlib import Path

from langchain_core.documents import Document

from raglab.composition.config import LocalIndexConfig, load_config
from raglab.composition.indexes import build_index

EXAMPLE_DIR = Path(__file__).resolve().parent

CONFIG_FILES = (
    "cendara_bm25.yaml",
    "cendara_bge_m3.yaml",
)


def load_documents(path: Path) -> list[Document]:
    """Load prepared chunks as LangChain documents."""

    chunks = json.loads(path.read_text(encoding="utf-8"))

    if not chunks:
        raise ValueError(f"No chunks found in {path}")

    return [
        Document(
            page_content=chunk["text"],
            metadata={"source": chunk["source"]},
        )
        for chunk in chunks
    ]


def build_retrieval_index(config_path: Path) -> None:
    """Build the retrieval index defined in an experiment configuration."""

    # 1. Load configuration and resolve paths
    config = load_config(
        config_path,
        project_root=EXAMPLE_DIR,
    )

    index = config.retriever.index

    if not isinstance(index, LocalIndexConfig):
        raise ValueError("Only local indexes are supported in this example")

    # 2. Skip an existing index
    if index.path.exists():
        print(f"Skipping {config_path.name}: {index.path} already exists")
        return

    # 3. Load the prepared Cendara chunks
    documents = load_documents(config.data.chunks)

    # 4. Build the index using the existing RAGLab component
    print(f"Building index from {len(documents)} chunks: {config_path.name}")

    build_index(
        config=config.retriever,
        documents=documents,
    )

    print(f"Index saved to: {index.path}")


if __name__ == "__main__":
    for filename in CONFIG_FILES:
        build_retrieval_index(EXAMPLE_DIR / filename)
