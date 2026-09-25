from raglab.infrastructure.generation.prompt import GenerationPrompt


def test_render_user_prompt() -> None:
    prompt = GenerationPrompt(
        system="System prompt.",
        user_template=("Context:\n{context}\n\nQuestion:\n{query}"),
    )

    result = prompt.render_user_prompt(
        query="Example question?",
        context="Example context.",
    )

    assert result == ("Context:\nExample context.\n\nQuestion:\nExample question?")
