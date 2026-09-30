"""Compare Cendara retrievers on the same successfully evaluated queries."""

import json
from pathlib import Path
from statistics import fmean
from typing import Any

EXAMPLE_DIR = Path(__file__).resolve().parent
REPORT_DIR = EXAMPLE_DIR / "artifacts" / "experiments"

REPORT_FILES = {
    "BM25S": REPORT_DIR / "cendara_bm25_evaluation.json",
    "BGE-M3": REPORT_DIR / "cendara_bge_m3_evaluation.json",
}

METRIC_NAMES = ("recall@1", "recall@5", "mrr")


def load_report(path: Path) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    """Load a completed evaluation report and index results by query."""

    report = json.loads(path.read_text(encoding="utf-8"))

    if report["status"] not in (
        "completed",
        "completed_with_judge_failures",
    ):
        raise ValueError(f"Experiment is not completed: {path}")

    results = {result["query_index"]: result for result in report["results"]}

    if len(results) != report["queries_total"]:
        raise ValueError(f"Missing or duplicate query results: {path}")

    if set(results) != set(range(report["queries_total"])):
        raise ValueError(f"Unexpected query indices: {path}")

    return report, results


def compare() -> None:
    """Calculate paired metrics and save the comparison report."""

    # 1. Load both evaluation reports
    bm25_report, bm25_results = load_report(REPORT_FILES["BM25S"])
    bge_report, bge_results = load_report(REPORT_FILES["BGE-M3"])

    # 2. Check that both experiments evaluated the same benchmark
    if bm25_report["queries_total"] != bge_report["queries_total"]:
        raise ValueError("Experiments have different benchmark sizes")

    for index in bm25_results:
        if bm25_results[index]["sample"] != bge_results[index]["sample"]:
            raise ValueError(f"Benchmark samples differ at query {index}")

    for report in (bm25_report, bge_report):
        if set(report["metric_names"]) != set(METRIC_NAMES):
            raise ValueError("Unexpected evaluation metrics")

    # 3. Select queries successfully evaluated by both retrievers
    paired_indices = sorted(
        index
        for index in bm25_results
        if bm25_results[index]["status"] == "success"
        and bge_results[index]["status"] == "success"
    )

    if not paired_indices:
        raise ValueError("No successfully evaluated queries in common")

    # 4. Calculate averages on the same query indices
    results_by_retriever = {
        "BM25S": bm25_results,
        "BGE-M3": bge_results,
    }

    comparison: dict[str, Any] = {
        "queries_total": bm25_report["queries_total"],
        "paired_queries": len(paired_indices),
        "excluded_query_indices": sorted(set(bm25_results) - set(paired_indices)),
        "retrievers": {},
    }

    for name, results in results_by_retriever.items():
        successful = sum(result["status"] == "success" for result in results.values())

        averages = {
            metric: fmean(results[index]["metrics"][metric] for index in paired_indices)
            for metric in METRIC_NAMES
        }

        comparison["retrievers"][name] = {
            "successful_queries": successful,
            "judge_failures": len(results) - successful,
            "paired_metrics": averages,
        }

    # 5. Save an aggregate report without credentials or document text
    output_path = REPORT_DIR / "cendara_comparison.json"

    output_path.write_text(
        json.dumps(comparison, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    # 6. Print the paired comparison
    print(f"Total queries: {comparison['queries_total']}")
    print(f"Paired queries: {comparison['paired_queries']}")
    print(f"Excluded indices: {comparison['excluded_query_indices']}")
    print()

    print(f"{'Metric':<12} {'BM25S':>10} {'BGE-M3':>10}")

    for metric in METRIC_NAMES:
        bm25 = comparison["retrievers"]["BM25S"]["paired_metrics"][metric]
        bge = comparison["retrievers"]["BGE-M3"]["paired_metrics"][metric]

        print(f"{metric:<12} {bm25:>10.4f} {bge:>10.4f}")

    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    compare()
