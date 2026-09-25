from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationPrompt:
    """Define prompts used for generation.

    Templates use Python str.format syntax: {query}, {context}.
    If a template needs literal braces (e.g. JSON), escape them
    as {{ and }}.
    """

    system: str
    user_template: str

    def render_user_prompt(
        self,
        query: str,
        context: str,
    ) -> str:
        """Render the user prompt."""

        return self.user_template.format(
            query=query,
            context=context,
        )


DEFAULT_RAG_PROMPT = GenerationPrompt(
    system=(
        "Answer the question using only the provided context. "
        "If the context does not contain enough information, "
        "say that you do not know."
    ),
    user_template=("Context:\n{context}\n\nQuestion:\n{query}"),
)
