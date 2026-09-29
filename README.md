# RAGLab

RAGLab is a configurable Python project for experimenting with Retrieval-Augmented Generation (RAG) and evaluating
retrieval strategies on question-answering benchmarks.

The project separates retrieval, generation and relevance evaluation into interchangeable components. Pipelines are
assembled from YAML configurations, making it possible to compare retrievers, embedding models and LLMs without changing
the core application logic

## Features

- **Configurable retrieval:** BM25S lexical search and vector search using Hugging Face embeddings and Chroma
- **Pluggable LLMs:** GigaChat integration for generation and relevance evaluation, with support for adding other
  providers
- **Retrieval evaluation:** Recall@k and MRR calculated against benchmark supporting facts
- **Relevance judges:** LLM-as-a-Judge and deterministic substring matching
- **Resumable experiments:** JSON checkpoints saved after each query, with support for retrying failed Judge evaluations
- **CLI and YAML configuration:** Run experiments without modifying application code

## Architecture

RAGLab uses a layered architecture with ports separating application logic from external libraries and services.

| Module            | Responsibility                                                         |
|-------------------|------------------------------------------------------------------------|
| `application/`    | RAG services, evaluation workflows, models and metrics                 |
| `ports/`          | Interfaces for retrievers, generators and relevance judges             |
| `infrastructure/` | BM25S, Chroma, Hugging Face, GigaChat and data storage implementations |
| `composition/`    | Configuration loading and pipeline construction                        |
| `presentation/`   | Command-line interface                                                 |
| `workflows/`      | High-level experiment execution and checkpoint handling                |
| `example/`        | Reproducible Cendara University retrieval experiment                   |
| `tests/`          | Unit and integration tests                                             |

LangChain is used for integrations, while RAGLab's own interfaces keep the application layer independent of particular
retrieval or LLM implementations.

## Installation

Requires Python 3.12.

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/OOLebedenko/raglab.git
cd raglab

python3.12 -m venv .venv
source .venv/bin/activate
```

Install the dependencies for your use case:

**Base installation**

```bash
python -m pip install -e .
```

**With Cendara example dependencies**

```bash
python -m pip install -e ".[example]"
```

**For development (tests, linting and type checking)**

```bash
python -m pip install -e ".[dev]"
```

## Usage

RAGLab provides a command-line interface for running configurable pipelines and retrieval evaluations

```bash
raglab --help
```

Experiment settings are defined in YAML files, including the dataset paths, retriever, embedding model, relevance Judge
and evaluation metrics.

For example, after preparing the Cendara dataset and building its indexes:

```bash
cd example

raglab evaluate -c cendara_bm25.yaml --project-root .
raglab evaluate -c cendara_bge_m3.yaml --project-root .
```

Evaluation reports are saved as JSON. Interrupted runs can resume from their checkpoints, and `--retry-failed` retries
queries that previously failed Judge validation.

See the [Cendara example](example/README.md) for complete installation, configuration and reproduction instructions

## Example: Cendara University

The included example uses the Cendara University subset
of [RAG-Multi-Corpus](https://github.com/udayallu/RAG-Multi-Corpus), with pre-generated chunks, benchmark questions and
gold supporting facts.

It compares BM25S and BGE-M3 using GigaChat-2 as an LLM-as-a-Judge, alongside a BM25S substring baseline.

The paired evaluation covers 186 successfully judged questions out of 187.

| Metric   | BM25S + GigaChat-2 | BGE-M3 + GigaChat-2 |
|----------|-------------------:|--------------------:|
| Recall@1 |               0.45 |                0.65 |
| Recall@5 |               0.78 |                0.88 |
| MRR      |               0.60 |                0.78 |

![Cendara retrieval comparison](example/artifacts/experiments/cendara_comparison.png)

BGE-M3 achieved higher scores across all three metrics in this experiment. The substring baseline also demonstrates how
the choice of relevance Judge affects the evaluation results.

Detailed results, experiment artifacts and known LLM-as-a-Judge limitations are documented in
the [example README](example/README.md).

## Extending RAGLab

Retrievers, generators and relevance judges are defined through separate interfaces. To introduce a new implementation,
add the corresponding infrastructure adapter and register it in the composition layer.

This allows experiments to switch between supported components through YAML configuration while keeping the evaluation
workflow and metrics unchanged
