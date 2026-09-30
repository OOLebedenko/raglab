from pathlib import Path

from dotenv import load_dotenv

from raglab.application.rag_service import RagService
from raglab.composition.config.loader import load_production_config
from raglab.composition.generators import build_generator
from raglab.composition.retrievers import build_retriever


def run_ask(
    config_path: Path,
    project_root: Path,
    query: str,
) -> str:
    """Generate an answer using the production configuration"""
    load_dotenv(project_root.resolve() / ".env")

    # 1. Load the production configuration
    config = load_production_config(
        config_path,
        project_root=project_root,
    )

    # 2. Connect to the existing index and create the retriever
    retriever = build_retriever(
        config.retriever,
        config.retrieval_policy,
    )

    # 3. Build the configured language model and generator
    generator = build_generator(config.generator)

    # 4. Assemble the RAG service
    service = RagService(
        retriever=retriever,
        generator=generator,
    )

    # 5. Retrieve context and generate an answer
    result = service.ask(query)

    # 6. Return the generated answer
    return result.answer
