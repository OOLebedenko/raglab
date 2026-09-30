# Architecture decisions

This document describes the main architectural decisions for RAGLab.

The goal is to preserve the reasoning behind the project structure without duplicating implementation details that already live in code.

## Layers

RAGLab is divided into four main layers:

- `application` — application scenarios and contracts.
  Contains services such as `RagService` and `EvaluationService`, application models, and ports used by these services.

- `infrastructure` — concrete implementations of external and technical details.
  Contains retrieval strategies, embedding implementations, generators, and data loaders.

- `composition` — object graph construction.
  Reads configuration, resolves concrete strategies, creates infrastructure components, and assembles application services.

- `presentation` — external interface to the application.
  The first interface is a Typer CLI.

The intended dependency direction is:

```text
presentation → composition → application → ports
                                   ↑
                            infrastructure
```
