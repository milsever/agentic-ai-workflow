from __future__ import annotations

from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_assistant.rag_graph import build_rag_graph

app = FastAPI(
    title="Enterprise RAG Knowledge Assistant",
    description="REST API for the LangGraph-based document question-answering workflow.",
    version="0.1.0",
)


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Question to answer using the indexed documents.",
        examples=["Akbank Mobil üzerinden EFT ücreti nedir?"],
    )


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[dict[str, Any]] = Field(default_factory=list)


@lru_cache(maxsize=1)
def get_rag_graph():
    """Create the graph once per API process, not once per request."""
    return build_rag_graph()


@app.get("/health", tags=["Health"])
def health() -> dict[str, str]:
    """Basic process health check; does not call the LLM."""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse, tags=["RAG"])
def ask(request: AskRequest) -> AskResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question must not be blank.")

    try:
        result = get_rag_graph().invoke(
            {
                "question": question,
                "context": "",
                "sources": [],
                "answer": "",
            }
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="The RAG workflow failed. Check the application logs.",
        ) from exc

    answer = result.get("answer", "")
    sources = result.get("sources", [])
    if not isinstance(answer, str):
        answer = str(answer)
    if not isinstance(sources, list):
        sources = []

    return AskResponse(question=question, answer=answer, sources=sources)
