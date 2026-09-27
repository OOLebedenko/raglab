import json
from pathlib import Path

import pytest
import yaml
from langchain_core.documents import Document

from raglab.application.models import QueryRelevance, SupportingFact
from raglab.composition.config import load_config
from raglab.composition.indexes import build_index
from raglab.composition.judges import build_judge
from raglab.composition.metrics import build_metrics
from raglab.composition.retrievers import build_retriever


@pytest.fixture
def chunks_path(tmp_path: Path) -> Path:
    """Create a temporary chunks file."""

    path = tmp_path / "chunks.json"

    chunks = [
        {
            "text": "Quantum spin resonance in magnetic fields",
            "source": "physics.md",
        },
        {
            "text": "Database transactions and isolation levels",
            "source": "database.md",
        },
        {
            "text": "Web applications use HTTP for communication",
            "source": "web.md",
        },
    ]

    path.write_text(
        json.dumps(chunks, indent=2),
        encoding="utf-8",
    )

    return path


@pytest.fixture
def benchmark_path(tmp_path: Path) -> Path:
    """Create a temporary benchmark file."""

    path = tmp_path / "benchmark.json"

    benchmark = [
        {
            "query": "quantum resonance",
            "supporting_facts": [
                {
                    "text": "Quantum spin resonance in magnetic fields",
                    "source": "physics.md",
                },
            ],
        },
    ]

    path.write_text(
        json.dumps(benchmark, indent=2),
        encoding="utf-8",
    )

    return path


@pytest.fixture
def config_path(
    tmp_path: Path,
    chunks_path: Path,
    benchmark_path: Path,
) -> Path:
    """Create a temporary experiment configuration."""

    path = tmp_path / "experiment.yaml"

    config = {
        "data": {
            "chunks": chunks_path.name,
            "benchmark": benchmark_path.name,
        },
        "retriever": {
            "type": "lexical",
            "index": {
                "type": "bm25s",
                "location": "local",
                "path": "bm25",
            },
        },
        "retrieval_policy": {"top_k": 1},
        "evaluation": {
            "judge": {"type": "substring"},
            "metrics": [
                {"type": "recall_at_k", "k": 1},
                {"type": "reciprocal_rank"},
            ],
        },
    }

    path.write_text(
        yaml.safe_dump(config),
        encoding="utf-8",
    )

    return path


def test_retrieval_evaluation_pipeline(config_path: Path) -> None:
    """Run retrieval and evaluation from configuration and files."""

    # 1. Load the experiment configuration and resolve data paths
    config = load_config(
        config_path,
        project_root=config_path.parent,
    )

    # 2. Read chunks and benchmark data from the configured files
    chunks = json.loads(config.data.chunks.read_text(encoding="utf-8"))
    benchmark = json.loads(config.data.benchmark.read_text(encoding="utf-8"))

    # 3. Convert chunks into LangChain documents
    documents = [
        Document(
            page_content=chunk["text"],
            metadata={"source": chunk["source"]},
        )
        for chunk in chunks
    ]

    # 4. Prepare the query and its gold supporting facts
    # The gold dataset contains a list of queries with their correct answers
    # For this test, we take only the first entry
    sample = benchmark[0]

    gold = [
        SupportingFact(
            text=fact["text"],
            source=fact["source"],
        )
        for fact in sample["supporting_facts"]
    ]

    # 5. Build the index and create the retriever from configuration
    build_index(config.retriever, documents)

    retriever = build_retriever(
        config.retriever,
        config.retrieval_policy,
    )

    # 6. Perform retrieval and verify the adapted results
    retrieved = retriever.retrieve(sample["query"])

    assert len(retrieved) == 1
    assert retrieved[0].text == chunks[0]["text"]
    assert retrieved[0].source == chunks[0]["source"]

    # 7. Build the relevance judge and match results against gold data
    judge = build_judge(config.evaluation.judge)
    matches = judge.find_matches(retrieved, gold)

    relevance_index = QueryRelevance(
        matches=matches,
        gold_count=len(gold),
    )

    # 8. Build and calculate the configured retrieval metrics
    metrics = build_metrics(config.evaluation.metrics)

    results = {metric.name: metric.calculate(relevance_index) for metric in metrics}

    # 9. Verify the expected evaluation results
    assert results == {
        "recall@1": 1.0,
        "mrr": 1.0,
    }
