"""Prepare the Cendara University dataset for RAGLab."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

EXAMPLE_DIR = Path(__file__).resolve().parent
DATA_DIR = EXAMPLE_DIR / "data"
DEFAULT_DATASET_ROOT = DATA_DIR / "upstream" / "RAG-Multi-Corpus"


def prepare(dataset_root: Path) -> None:
    """Convert the authors' chunks and queries into RAGLab JSON files."""

    references = dataset_root / "bechmark" / "bechmark-agentic-references"

    chunks_path = references / "parsed-chunks" / "cendara_prompt_chunks.json"
    queries_path = references / "Dataset categories - queries.csv"

    for path in (chunks_path, queries_path):
        if not path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {path}")

    # 1. Load the authors' chunks without changing their text or order
    chunks: list[dict[str, str]] = []

    with chunks_path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            document = json.loads(line)

            for text in document["chunks"]:
                chunks.append(
                    {
                        "text": text,
                        "source": document["file_name"],
                    }
                )

    # 2. Select Cendara queries and their supporting facts
    benchmark: list[dict[str, Any]] = []

    with queries_path.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            if row["Enterprise Name"].strip() != "Cendara University":
                continue

            facts = json.loads(row["Supporting Facts"])

            benchmark.append(
                {
                    "query": row["Query"],
                    "supporting_facts": [
                        {
                            "text": fact["text"],
                            "source": fact["filename"],
                        }
                        for fact in facts
                    ],
                }
            )

    if not chunks or not benchmark:
        raise ValueError("Cendara chunks or benchmark queries are missing")

    # 3. Validate supporting fact sources
    chunk_sources = {chunk["source"] for chunk in chunks}

    fact_sources = {
        fact["source"] for sample in benchmark for fact in sample["supporting_facts"]
    }

    missing_sources = fact_sources - chunk_sources

    if missing_sources:
        raise ValueError(f"Missing supporting fact sources: {sorted(missing_sources)}")

    # 4. Save the prepared dataset inside the example directory
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    (DATA_DIR / "chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    (DATA_DIR / "benchmark.json").write_text(
        json.dumps(benchmark, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Chunks: {len(chunks)}")
    print(f"Queries: {len(benchmark)}")
    print(f"Saved to: {DATA_DIR}")


def main() -> None:
    """Parse arguments and prepare the dataset."""

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Path to the cloned RAG-Multi-Corpus repository",
    )

    args = parser.parse_args()
    prepare(args.dataset_root.resolve())


if __name__ == "__main__":
    main()
