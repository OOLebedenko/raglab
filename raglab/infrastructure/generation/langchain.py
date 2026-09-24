from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from raglab.application.models import RetrievedChunk
from raglab.infrastructure.generation.prompt import GenerationPrompt


def format_context(
    chunks: list[RetrievedChunk],
) -> str:
    """Format retrieved chunks as generation context."""

    return "\n\n".join(chunk.text for chunk in chunks)


class LangChainGeneratorAdapter:
    """Adapt a LangChain chat model to the RAGLab generator contract."""

    def __init__(
        self,
        model: BaseChatModel,
        prompt: GenerationPrompt,
    ) -> None:
        self._model = model
        self._prompt = prompt

    def generate(
        self,
        query: str,
        context: list[RetrievedChunk],
    ) -> str:
        """Generate an answer from a query and retrieved context."""

        context_text = format_context(context)

        user_prompt = self._prompt.render_user_prompt(
            query=query,
            context=context_text,
        )

        messages = [
            SystemMessage(
                content=self._prompt.system,
            ),
            HumanMessage(
                content=user_prompt,
            ),
        ]

        response = self._model.invoke(messages)
        content = response.content

        if not isinstance(content, str):
            raise ValueError("LangChain chat model must return string content.")

        return content
