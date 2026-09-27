from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from langchain_core.retrievers import BaseRetriever

from raglab.composition.config import (
    EmbeddingConfig,
    LexicalRetrieverConfig,
    LocalIndexConfig,
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
        embedding=EmbeddingConfig(
            type="huggingface",
            model_name="test-model",
        ),
        index=LocalIndexConfig(
            type="chroma",
            location="local",
            path=tmp_path,
            options={"collection_name": "test_collection"},
        ),
    )
    policy = RetrievalPolicyConfig(top_k=7)

    retriever = Mock(spec=BaseRetriever)
    mock_builder = Mock(return_value=retriever)

    with (
        patch("raglab.composition.retrievers.build_embeddings") as mock_embeddings,
        patch(
            "raglab.composition.retrievers._VECTOR_BUILDERS",
            {"chroma": mock_builder},
        ),
    ):
        result = build_vector_retriever(config, policy)

    mock_embeddings.assert_called_once_with(config.embedding)
    assert isinstance(config.index, LocalIndexConfig)

    mock_builder.assert_called_once_with(
        config.index.path,
        mock_embeddings.return_value,
        config.index.options,
        7,
    )

    assert result is retriever


def test_build_lexical_retriever(tmp_path: Path) -> None:
    config = LexicalRetrieverConfig(
        type="lexical",
        index=LocalIndexConfig(
            type="bm25s",
            location="local",
            path=tmp_path,
        ),
    )
    policy = RetrievalPolicyConfig(top_k=7)

    mock_factory = Mock()

    with patch(
        "raglab.composition.retrievers._LEXICAL_BUILDERS",
        {"bm25s": mock_factory},
    ):
        result = build_lexical_retriever(config, policy)

    mock_factory.assert_called_once_with(tmp_path, 7)
    assert result is mock_factory.return_value


def test_build_lexical_retriever_rejects_unknown_implementation(
    tmp_path: Path,
) -> None:
    config = LexicalRetrieverConfig(
        type="lexical",
        index=LocalIndexConfig(
            type="unknown",
            location="local",
            path=tmp_path,
        ),
    )
    policy = RetrievalPolicyConfig(top_k=5)

    with pytest.raises(ValueError, match="Unsupported lexical retriever"):
        build_lexical_retriever(config, policy)
