
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from rag_assistant.rag_graph import build_rag_graph


def make_document(
    content="Transfer fee is 5 TL.",
    source="fees.pdf",
    page=2,
):
    return Document(
        page_content=content,
        metadata={"source": source, "page": page},
    )


@patch("rag_assistant.rag_graph.get_chat_model")
@patch("rag_assistant.rag_graph.HybridRetriever")
def test_rag_graph_retrieves_documents_and_generates_answer(
    mock_retriever_class,
    mock_get_chat_model,
):
    retriever = MagicMock()
    retriever.invoke.return_value = [make_document()]
    mock_retriever_class.return_value = retriever

    model = MagicMock()
    model.invoke.return_value = MagicMock(
        content="The transfer fee is 5 TL [S1]."
    )
    mock_get_chat_model.return_value = model

    graph = build_rag_graph()
    result = graph.invoke({
        "question": "What is the transfer fee?",
        "context": "",
        "sources": [],
        "answer": "",
    })

    assert result["answer"] == "The transfer fee is 5 TL [S1]."
    assert "[S1]" in result["context"]
    assert "Transfer fee is 5 TL." in result["context"]
    assert result["sources"] == [{
        "id": "S1",
        "source": "fees.pdf",
        "page": 2,
        "content": "Transfer fee is 5 TL.",
    }]

    retriever.invoke.assert_called_once_with(
        "What is the transfer fee?"
    )
    model.invoke.assert_called_once()


@patch("rag_assistant.rag_graph.get_chat_model")
@patch("rag_assistant.rag_graph.HybridRetriever")
def test_rag_graph_does_not_call_llm_when_context_is_empty(
    mock_retriever_class,
    mock_get_chat_model,
):
    retriever = MagicMock()
    retriever.invoke.return_value = []
    mock_retriever_class.return_value = retriever

    model = MagicMock()
    mock_get_chat_model.return_value = model

    graph = build_rag_graph()
    result = graph.invoke({
        "question": "Unknown topic",
        "context": "",
        "sources": [],
        "answer": "",
    })

    assert result["context"] == ""
    assert result["sources"] == []
    assert result["answer"] == (
        "I could not find relevant information "
        "in the indexed documents."
    )
    model.invoke.assert_not_called()
