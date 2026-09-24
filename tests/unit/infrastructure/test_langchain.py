from langchain_core.documents import Document

from raglab.infrastructure.retrieval.langchain import LangChainRetrieverAdapter


class FakeLangChainRetriever:
    """Return predefined LangChain documents."""

    def __init__(
        self,
        documents: list[Document],
    ) -> None:
        self._documents = documents

    def invoke(
        self,
        input: str,
    ) -> list[Document]:
        return self._documents


def test_retrieve_maps_document_with_score() -> None:
    document = Document(
        page_content="First chunk.",
        metadata={
            "source": "document.txt",
            "score": 0.91,
            "page": 1,
        },
    )

    retriever = FakeLangChainRetriever(
        documents=[document],
    )
    adapter = LangChainRetrieverAdapter(retriever)

    result = adapter.retrieve("Example query")

    assert result[0].text == "First chunk."
    assert result[0].source == "document.txt"
    assert result[0].score == 0.91
    assert result[0].metadata["page"] == 1


def test_retrieve_maps_document_without_score() -> None:
    document = Document(
        page_content="Second chunk.",
        metadata={
            "source": "document.txt",
        },
    )

    retriever = FakeLangChainRetriever(
        documents=[document],
    )
    adapter = LangChainRetrieverAdapter(retriever)

    result = adapter.retrieve("Example query")

    assert result[0].score is None
