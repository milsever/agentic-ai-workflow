import os
import logging

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from rag_assistant.multi_agent_graph import build_multi_agent_graph

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_VERBOSITY"] = "error"

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.ERROR)

    console = Console()
    graph = build_multi_agent_graph()

    console.print(
        Panel(
            "[bold cyan]Enterprise Multi-agent Knowledge Assistant[/bold cyan]\n"
            "Enterprise Knowledge | Loan Applications | Multi-agent Workflow\n\n"
            "[dim]Type 'q', 'quit', or 'exit' to stop.[/dim]",
            title="Agentic AI Workflow",
            border_style="blue",
        )
    )

    while True:
        try:
            question = console.input("[bold green]> [/bold green]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]Assistant stopped.[/yellow]")
            break

        if question.lower() in {"q", "quit", "exit"}:
            console.print("[yellow]Assistant stopped.[/yellow]")
            break

        if not question:
            console.print("[yellow]Please enter a question.[/yellow]")
            continue

        try:
            # with console.status(
            #         "[cyan]Orchestrator and specialists are working...[/cyan]",
            #         spinner="dots",
            # ):
            result = graph.invoke({"messages": [("user", question)]})

            routes = result.get("routes", [])

            if routes:
                specialist_names = {
                    "rag_agent": "RAG Specialist",
                    "application_agent": "Application Specialist",
                }
                selected = [
                    specialist_names.get(route, route)
                    for route in routes
                ]
                console.print(
                    "[dim]Selected specialists: "
                    + ", ".join(selected)
                    + "[/dim]"
                )
            else:
                console.print(
                    "[dim]No specialist was selected.[/dim]"
                )

            messages = result.get("messages", [])
            if not messages:
                console.print(
                    "[red]The workflow returned no messages.[/red]"
                )
                continue

            answer = messages[-1].content

            if isinstance(answer, list):
                answer = "\n".join(
                    str(block.get("text", block))
                    if isinstance(block, dict)
                    else str(block)
                    for block in answer
                )

            console.print(
                Panel(
                    Markdown(str(answer)),
                    title="Assistant Response",
                    border_style="blue",
                )
            )

        except Exception:
            logging.exception("Multi-agent workflow failed")
            console.print(
                "[red]An error occurred while processing the request. "
                "Check the terminal logs for details.[/red]"
            )

if __name__ == "__main__":
    main()
