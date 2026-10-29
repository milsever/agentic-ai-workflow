import json
import logging
from time import perf_counter

from langchain_core.tools import tool

from rag_assistant.rag_graph import build_rag_graph

logger = logging.getLogger(__name__)

def create_rag_tool(graph=None):
    """Create a LangChain tool backed by the existing RAG graph."""

    if graph is None:
        graph = build_rag_graph()

    @tool(
        "search_rag_knowledge",
        description=(
                "Search the enterprise knowledge documents to answer "
                "questions about banking products, fees, banking and financial-sector laws, "
                "banking regulations, and company research. Use this tool when the answer "
                "requires information from the indexed documents. "
                "Returns a document-grounded answer and its sources."
        ),
    )
    def search_rag_knowledge(question: str) -> str:
        """Search indexed documents using the user's question."""

        start_time = perf_counter()
        logger.info("RAG search started")

        try:
            result = graph.invoke({
                "question": question,
                "context": "",
                "sources": [],
                "answer": "",
            })

            answer = result.get("answer", "")
            sources = result.get("sources", [])

            if not answer:
                logger.warning("RAG search returned an empty answer")
            else:
                logger.info(
                    "RAG search completed in %.3f seconds",
                    perf_counter() - start_time,
                )

            return json.dumps(
                {
                    "answer": answer,
                    "sources": sources,
                },
                ensure_ascii=False,
            )

        except Exception as exc:
            logger.exception("RAG search failed")

            return json.dumps(
                {
                    "error": "rag_search_failed",
                    "message": str(exc),
                },
                ensure_ascii=False,
            )

    return search_rag_knowledge
