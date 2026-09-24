from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from raglab.application.models import RetrievedChunk


def to_retrieved_chunk(
    document: Document,
) -> RetrievedChunk:
    """Convert a LangChain document to a RAGLab retrieved chunk."""

    source = document.metadata.get("source")

    if not isinstance(source, str):
        raise ValueError("LangChain document metadata must contain a string 'source'.")

    raw_score = document.metadata.get("score")

    if raw_score is None:
        score = None
    elif isinstance(raw_score, (int, float)) and not isinstance(raw_score, bool):
        score = float(raw_score)
    else:
        raise ValueError("LangChain document metadata 'score' must be numeric.")

    return RetrievedChunk(
        text=document.page_content,
        source=source,
        score=score,
        metadata=dict(document.metadata),
    )


class LangChainRetrieverAdapter:
    """Adapt a LangChain retriever to the RAGLab retriever contract."""

    def __init__(
        self,
        retriever: BaseRetriever,
    ) -> None:
        self._retriever = retriever

    def retrieve(
        self,
        query: str,
    ) -> list[RetrievedChunk]:
        """Retrieve chunks relevant to a query."""

        documents = self._retriever.invoke(query)

        return [to_retrieved_chunk(document) for document in documents]
