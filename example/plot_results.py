"""Plot Cendara retrieval results with LLM and substring judges."""

import json
from pathlib import Path
from statistics import fmean

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

EXAMPLE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = EXAMPLE_DIR / "artifacts" / "experiments"

REPORT_PATH = ARTIFACTS_DIR / "cendara_comparison.json"
SUBSTRING_PATH = ARTIFACTS_DIR / "cendara_bm25_substring.json"
PLOT_PATH = ARTIFACTS_DIR / "cendara_comparison.png"

METRICS = {
    "recall@1": "Recall@1",
    "recall@5": "Recall@5",
    "mrr": "MRR",
}


def plot_results() -> None:
    """Build a grouped bar chart on the same benchmark queries."""

    # 1. Load the LLM and substring evaluation results
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    substring_report = json.loads(SUBSTRING_PATH.read_text(encoding="utf-8"))

    bm25 = report["retrievers"]["BM25S"]["paired_metrics"]
    bge_m3 = report["retrievers"]["BGE-M3"]["paired_metrics"]

    # 2. Select the same queries used in the paired LLM comparison
    paired_indices = set(range(report["queries_total"])) - set(
        report["excluded_query_indices"]
    )

    if len(paired_indices) != report["paired_queries"]:
        raise ValueError("Invalid paired query indices")

    if substring_report["queries_total"] != report["queries_total"]:
        raise ValueError("Substring and LLM benchmark sizes differ")

    substring_results = {
        result["query_index"]: result for result in substring_report["results"]
    }

    if len(substring_results) != len(substring_report["results"]):
        raise ValueError("Duplicate substring query indices")

    if not paired_indices.issubset(substring_results):
        raise ValueError("Missing substring results for paired queries")

    substring = {
        metric: fmean(
            substring_results[index]["metrics"][metric]
            for index in sorted(paired_indices)
        )
        for metric in METRICS
    }

    # 3. Prepare the chart data
    labels = list(METRICS.values())

    substring_scores = [substring[name] for name in METRICS]
    bm25_scores = [bm25[name] for name in METRICS]
    bge_scores = [bge_m3[name] for name in METRICS]

    positions = list(range(len(METRICS)))
    width = 0.23

    fig, ax = plt.subplots(figsize=(12, 7))

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # 4. Plot three evaluation variants for each metric
    substring_bars = ax.bar(
        [position - width for position in positions],
        substring_scores,
        width,
        label="BM25S + Substring",
        color="#A5BBDD",
        zorder=3,
    )

    bm25_bars = ax.bar(
        positions,
        bm25_scores,
        width,
        label="BM25S + GigaChat-2",
        color="#6489B6",
        zorder=3,
    )

    bge_bars = ax.bar(
        [position + width for position in positions],
        bge_scores,
        width,
        label="BGE-M3 + GigaChat-2",
        color="#55A596",
        zorder=3,
    )

    # 5. Display values rounded to two decimal places
    for bars in (substring_bars, bm25_bars, bge_bars):
        ax.bar_label(
            bars,
            fmt="%.2f",
            padding=7,
            fontsize=15,
        )

    # 6. Format the axes and legend
    ax.set_xticks(positions, labels, fontsize=18)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score", fontsize=18, labelpad=12)

    ax.set_title(
        "Cendara University — Retrieval Evaluation",
        fontsize=24,
        fontweight="bold",
        pad=38,
    )

    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.grid(axis="y", alpha=0.2, zorder=0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.tick_params(
        axis="both",
        length=0,
        pad=10,
        labelsize=14,
    )

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=3,
        frameon=False,
        fontsize=16,
    )

    fig.text(
        0.5,
        0.01,
        f"{report['paired_queries']} shared queries"
        f" / {report['queries_total']} total · top-5",
        ha="center",
        fontsize=16,
        color="#777777",
    )

    # 7. Save the figure
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))

    fig.savefig(
        PLOT_PATH,
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    print(f"Plot saved to: {PLOT_PATH}")
    print(f"Paired queries: {len(paired_indices)}")
    print("BM25S + Substring:")

    for name, value in substring.items():
        print(f"  {name}: {value:.4f}")


if __name__ == "__main__":
    plot_results()
