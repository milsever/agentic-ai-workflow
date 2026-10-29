
import json
from unittest.mock import MagicMock, patch

from rag_assistant.tools.rag_tool import create_rag_tool


@patch("rag_assistant.tools.rag_tool.build_rag_graph")
def test_rag_tool_returns_answer_and_sources(mock_build_graph):
    graph = MagicMock()
    graph.invoke.return_value = {
        "answer": "The transfer fee is 5 TL [S1].",
        "sources": [
            {
                "id": "S1",
                "source": "fees.pdf",
                "page": 2,
                "content": "Transfer fee is 5 TL.",
            }
        ],
    }
    mock_build_graph.return_value = graph

    rag_tool = create_rag_tool()
    result = rag_tool.invoke({
        "question": "What is the transfer fee?"
    })

    output = json.loads(result)

    assert output["answer"] == "The transfer fee is 5 TL [S1]."
    assert output["sources"][0]["source"] == "fees.pdf"
    assert output["sources"][0]["page"] == 2

    graph.invoke.assert_called_once_with({
        "question": "What is the transfer fee?",
        "context": "",
        "sources": [],
        "answer": "",
    })
