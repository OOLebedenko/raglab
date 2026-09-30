from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from raglab.workflows.evaluation import run_evaluation
from raglab.workflows.generation import run_ask

ConfigPath = Annotated[
    Path,
    typer.Option(
        "--config",
        "-c",
        exists=True,
        dir_okay=False,
        resolve_path=True,
        help="Path to the experiment or production configuration",
    ),
]

ProjectRoot = Annotated[
    Path | None,
    typer.Option(
        "--project-root",
        exists=True,
        file_okay=False,
        resolve_path=True,
        help="Project root for resolving paths in the configuration",
    ),
]

app = typer.Typer(
    no_args_is_help=True,
    help="RAGLab command-line interface",
)

console = Console()


@app.callback()
def main() -> None:
    """RAGLab command-line interface"""


@app.command()
def evaluate(
    config: ConfigPath,
    project_root: ProjectRoot = None,
    retry_failed: Annotated[
        bool,
        typer.Option(
            "--retry-failed",
            help="Re-evaluate queries that previously failed during judging",
        ),
    ] = False,
) -> None:
    """Evaluate retrieval using the experiment configuration"""

    root = project_root or Path.cwd()

    with console.status("Running retrieval evaluation..."):
        summary = run_evaluation(
            config_path=config,
            project_root=root,
            retry_failed=retry_failed,
        )

    table = Table(title="Retrieval evaluation")
    table.add_column("Metric")
    table.add_column("Value", justify="right")

    table.add_row("Status", summary.status)
    table.add_row("Total queries", str(summary.queries_total))
    table.add_row("Evaluated", str(summary.queries_evaluated))
    table.add_row("Judge failures", str(summary.queries_failed))

    for name, value in summary.metrics.items():
        formatted = f"{value:.4f}" if value is not None else "N/A"
        table.add_row(name, formatted)

    console.print(table)


@app.command()
def ask(
    query: Annotated[
        str,
        typer.Argument(help="Question to answer"),
    ],
    config: ConfigPath,
    project_root: ProjectRoot = None,
) -> None:
    """Answer a question using the production configuration"""

    root = project_root or Path.cwd()

    with console.status("Generating answer..."):
        answer = run_ask(
            config_path=config,
            project_root=root,
            query=query,
        )

    console.print(answer)


if __name__ == "__main__":
    app()
