from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from raglab.composition.config import (
    EmbeddingConfig,
    LexicalRetrieverConfig,
    LocalVectorStoreConfig,
    RetrievalPolicyConfig,
    VectorRetrieverConfig,
)
from raglab.composition.retrievers import (
    build_lexical_retriever,
    build_vector_retriever,
)


def test_build_vector_retriever(tmp_path: Path) -> None:
    config = VectorRetrieverConfig(
        type="vector",
        policy=RetrievalPolicyConfig(top_k=7),
        embedding=EmbeddingConfig(
            type="huggingface",
            model_name="test-model",
        ),
        vector_store=LocalVectorStoreConfig(
            type="chroma",
            path=tmp_path,
            options={"collection_name": "test_collection"},
        ),
    )

    with (
        patch("raglab.composition.retrievers.build_embeddings") as mock_embeddings,
        patch("raglab.composition.retrievers.build_vector_store") as mock_store,
    ):
        retriever = Mock(spec=BaseRetriever)
        mock_store.return_value.as_retriever.return_value = retriever

        result = build_vector_retriever(config)

    mock_embeddings.assert_called_once_with(config.embedding)

    mock_store.assert_called_once_with(
        config.vector_store,
        mock_embeddings.return_value,
    )

    mock_store.return_value.as_retriever.assert_called_once_with(
        search_kwargs={"k": 7},
    )

    assert result is retriever


def test_build_lexical_retriever() -> None:
    config = LexicalRetrieverConfig(
        type="lexical",
        implementation="bm25",
        policy=RetrievalPolicyConfig(top_k=7),
    )
    documents = [
        Document(page_content="First chunk", metadata={"source": "a.md"}),
        Document(page_content="Second chunk", metadata={"source": "b.md"}),
    ]

    with patch(
        "raglab.composition.retrievers.BM25Retriever.from_documents"
    ) as mock_factory:
        result = build_lexical_retriever(config, documents)

    mock_factory.assert_called_once_with(documents, k=7)
    assert result is mock_factory.return_value


def test_build_lexical_retriever_rejects_unknown_implementation() -> None:
    config = LexicalRetrieverConfig(
        type="lexical",
        implementation="unknown",
        policy=RetrievalPolicyConfig(top_k=5),
    )

    with pytest.raises(ValueError, match="Unsupported lexical retriever"):
        build_lexical_retriever(
            config,
            [Document(page_content="Example")],
        )
