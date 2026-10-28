
from typing import TypedDict

from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END

from rag_assistant.llm.factory import get_chat_model
from rag_assistant.retrieval.hybrid_retriever import HybridRetriever


class RAGState(TypedDict):
    question: str
    context: str
    sources: list[dict]
    answer: str


def build_rag_graph():
    retriever = HybridRetriever(final_k=10)
    model = get_chat_model()

    def retrieve(state: RAGState) -> dict:
        documents = retriever.invoke(state["question"])

        context_parts = []
        sources = []

        for index, doc in enumerate(documents, start=1):
            source_id = f"S{index}"
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "Unknown")

            context_parts.append(
                f"[{source_id}] Source: {source}, "
                f"Page: {page}\n{doc.page_content}"
            )

            sources.append({
                "id": source_id,
                "source": source,
                "page": page,
                "content": doc.page_content,
            })

        return {
            "context": "\n\n".join(context_parts),
            "sources": sources,
        }

    def generate(state: RAGState) -> dict:
        if not state["context"].strip():
            return {
                "answer": (
                    "I could not find relevant information "
                    "in the indexed documents."
                )
            }

        response = model.invoke([
            (
                "system",
                """You are a document-grounded knowledge assistant. Your rules are:

                1. Answer the user's question using only the supplied context.
                2. Do not invent facts that are absent from the context.
                3. If the context is insufficient, clearly say so.
                4. Cite supporting passages using their source IDs, such as [S1].
                5. Distinguish facts from any uncertainty.
                6. Answer in the same language as the user's question.
                7. Do not ask follow-up questions, or make suggestions."""
            ),
            (
                "human",
                f"Question:\n{state['question']}\n\n"
                f"Retrieved context:\n{state['context']}"
            ),
        ])

        return {"answer": str(response.content)}

    builder = StateGraph(RAGState)

    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)

    return builder.compile()
