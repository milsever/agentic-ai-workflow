
from rag_assistant.rag_graph import build_rag_graph

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

def main() -> None:
    graph = build_rag_graph()
    console = Console()

    console.print("[blue]Enterprise RAG Knowledge Assistant:[/blue]")
    while True:
        question = input("\033[32m> \033[0m").strip()
        if question.lower() in {"q", "quit", "exit"}:
            break

        if not question:
            print("Please enter a question.")
            return

        result = graph.invoke({
            "question": question,
            "context": "",
            "sources": [],
            "answer": "",
        })

        # print("\n=== ANSWER ===")
        # print(result["answer"])
        panel_output = Panel(Markdown(result["answer"]),
                             title="=== Assistant Response ===", border_style="blue", )
        console.print(panel_output)

        console.print("[cyan]=== SOURCES ===[/cyan]")
        for source in result["sources"]:
            print(
                f"[{source['id']}] "
                f"{source['source']} | Page {source['page']}"
            )

if __name__ == "__main__":
    main()

