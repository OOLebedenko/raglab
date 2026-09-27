import json
from pathlib import Path

from langchain_core.documents import Document

from raglab.application.models import QueryRelevance, SupportingFact
from raglab.composition.config import LocalIndexConfig, load_config
from raglab.composition.indexes import build_index
from raglab.composition.judges import build_judge
from raglab.composition.metrics import build_metrics
from raglab.composition.retrievers import build_retriever


def run_evaluation(
    config_path: Path,
    project_root: Path,
) -> dict[str, float]:
    """Evaluate retrieval using the experiment configuration"""

    # 1. Load the experiment configuration
    config = load_config(
        config_path,
        project_root=project_root,
    )

    # 2. Load benchmark queries and their supporting facts
    benchmark = json.loads(config.data.benchmark.read_text(encoding="utf-8"))

    if not benchmark:
        raise ValueError("Cannot evaluate an empty benchmark")

    # 3. Build the index if it does not exist
    if not isinstance(config.retriever.index, LocalIndexConfig):
        raise NotImplementedError("Remote indexes are not implemented")

    if not config.retriever.index.path.exists():
        chunks = json.loads(config.data.chunks.read_text(encoding="utf-8"))

        documents = [
            Document(
                page_content=chunk["text"],
                metadata={"source": chunk["source"]},
            )
            for chunk in chunks
        ]

        build_index(config.retriever, documents)

    # 4. Assemble the retrieval and evaluation components
    retriever = build_retriever(
        config.retriever,
        config.retrieval_policy,
    )

    judge = build_judge(config.evaluation.judge)
    metrics = build_metrics(config.evaluation.metrics)

    # 5. Initialize metric accumulators
    totals = {metric.name: 0.0 for metric in metrics}

    # 6. Retrieve and evaluate results for every benchmark query
    for sample in benchmark:
        gold = [
            SupportingFact(
                text=fact["text"],
                source=fact["source"],
            )
            for fact in sample["supporting_facts"]
        ]

        retrieved = retriever.retrieve(sample["query"])

        matches = judge.find_matches(
            retrieved,
            gold,
        )

        relevance = QueryRelevance(
            matches=matches,
            gold_count=len(gold),
        )

        # 7. Calculate metrics for the current query
        for metric in metrics:
            totals[metric.name] += metric.calculate(relevance)

    # 8. Average metric values across all benchmark queries
    return {name: total / len(benchmark) for name, total in totals.items()}
