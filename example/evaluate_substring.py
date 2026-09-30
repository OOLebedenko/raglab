"""Evaluate saved BM25S results using SubstringRelevanceJudge."""

import json
from pathlib import Path
from statistics import fmean
from typing import Any

from raglab.application.models import QueryRelevance, RetrievedChunk, SupportingFact
from raglab.composition.config import load_config
from raglab.composition.metrics import build_metrics
from raglab.infrastructure.relevance.substring import SubstringRelevanceJudge

EXAMPLE_DIR = Path(__file__).resolve().parent
REPORT_DIR = EXAMPLE_DIR / "artifacts" / "experiments"

CONFIG_PATH = EXAMPLE_DIR / "cendara_bm25.yaml"
INPUT_PATH = REPORT_DIR / "cendara_bm25_evaluation.json"
OUTPUT_PATH = REPORT_DIR / "cendara_bm25_substring.json"


def evaluate_substring() -> None:
    """Evaluate saved BM25S retrieval results with substring matching."""

    # 1. Load the configuration and existing BM25S report
    config = load_config(
        CONFIG_PATH,
        project_root=EXAMPLE_DIR,
    )

    report = json.loads(INPUT_PATH.read_text(encoding="utf-8"))

    results_by_index = {result["query_index"]: result for result in report["results"]}

    if len(results_by_index) != len(report["results"]) or set(results_by_index) != set(
        range(report["queries_total"])
    ):
        raise ValueError("Missing or duplicate query results")

    # 2. Initialize the substring judge and evaluation metrics
    judge = SubstringRelevanceJudge()
    metrics = build_metrics(config.evaluation.metrics)

    # 3. Evaluate the saved retrieved chunks for every query
    results: list[dict[str, Any]] = []

    for query_index, saved in sorted(results_by_index.items()):
        sample = saved["sample"]

        retrieved = [
            RetrievedChunk(
                text=chunk["text"],
                source=chunk["source"],
            )
            for chunk in saved["retrieved_chunks"]
        ]

        gold = [
            SupportingFact(
                text=fact["text"],
                source=fact["source"],
            )
            for fact in sample["supporting_facts"]
        ]

        matches = judge.find_matches(
            retrieved=retrieved,
            gold=gold,
        )

        relevance = QueryRelevance(
            matches=matches,
            gold_count=len(gold),
        )

        scores = {metric.name: float(metric.calculate(relevance)) for metric in metrics}

        results.append(
            {
                "query_index": query_index,
                "matches": [list(pair) for pair in sorted(matches)],
                "metrics": scores,
            }
        )

    # 4. Calculate average metrics across all benchmark queries
    averages = {
        metric.name: fmean(result["metrics"][metric.name] for result in results)
        for metric in metrics
    }

    # 5. Save the substring evaluation results
    output = {
        "retriever": "BM25S",
        "judge": "substring",
        "queries_total": len(results),
        "average_metrics": averages,
        "results": results,
    }

    OUTPUT_PATH.write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    # 6. Print the evaluation summary
    print(f"Evaluated queries: {len(results)}")

    for name, value in averages.items():
        print(f"{name}: {value:.4f}")

    print(f"\nReport saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    evaluate_substring()
