from pathlib import Path
from unittest.mock import Mock, patch

from langchain_core.retrievers import BaseRetriever

from raglab.composition.config import (
    EmbeddingConfig,
    LocalVectorStoreConfig,
    RetrievalPolicyConfig,
    VectorRetrieverConfig,
)
from raglab.composition.retrievers import build_vector_retriever


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
