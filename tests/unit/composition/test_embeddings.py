from unittest.mock import patch

from raglab.composition.config import EmbeddingConfig
from raglab.composition.embeddings import build_embeddings


def test_build_huggingface_embeddings() -> None:
    """Pass the configured model name to HuggingFace."""

    config = EmbeddingConfig(
        type="huggingface",
        model_name="selected-embedding-model",
    )

    with patch(
        "raglab.composition.embeddings.HuggingFaceEmbeddings"
    ) as mock_embeddings:
        result = build_embeddings(config)

    mock_embeddings.assert_called_once_with(model_name=config.model_name)
    assert result is mock_embeddings.return_value
