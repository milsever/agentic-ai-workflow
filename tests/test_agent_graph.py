from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from rag_assistant.agent_graph import build_agent_graph

@tool
def fake_rag_tool(question: str) -> str:
    """Search the test knowledge base."""
    return (
        '{"answer": "Transfer fee is 5 TL [S1].", '
        '"sources": [{"source": "fees.pdf", "page": 2}]}'
    )

@tool
def fake_application_tool(application_id: str) -> str:
    """Look up a test loan application."""
    return '{"application_id": "APP-1002", "status": "under_review"}'

def test_agent_calls_rag_tool_and_returns_final_answer():
    model = MagicMock()
    model_with_tools = MagicMock()

    model.bind_tools.return_value = model_with_tools

    model_with_tools.invoke.side_effect = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "fake_rag_tool",
                    "args": {"question": "What is the transfer fee?"},
                    "id": "call_1",
                    "type": "tool_call",
                }
            ],
        ),
        AIMessage(
            content="The transfer fee is 5 TL [S1]."
        ),
    ]

    with (
        patch(
            "rag_assistant.agent_graph.get_chat_model",
            return_value=model,
        ),
        patch(
            "rag_assistant.agent_graph.create_rag_tool",
            return_value=fake_rag_tool,
        ),
        patch(
            "rag_assistant.agent_graph.create_application_status_tool",
            return_value=fake_application_tool,
        ),
    ):
        graph = build_agent_graph()

        result = graph.invoke({
            "messages": [
                ("user", "What is the transfer fee?")
            ]
        })

    messages = result["messages"]

    assert messages[-1].content == (
        "The transfer fee is 5 TL [S1]."
    )

    assert len(messages) == 4
    assert messages[1].tool_calls[0]["name"] == "fake_rag_tool"
    assert messages[2].type == "tool"
    assert "Transfer fee is 5 TL" in messages[2].content

    model.bind_tools.assert_called_once_with(
        [fake_rag_tool, fake_application_tool]
    )
    assert model_with_tools.invoke.call_count == 2

def test_agent_calls_application_tool_and_returns_final_answer():
    model = MagicMock()
    model_with_tools = MagicMock()
    model.bind_tools.return_value = model_with_tools

    model_with_tools.invoke.side_effect = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "fake_application_tool",
                    "args": {"application_id": "APP-1002"},
                    "id": "call_2",
                    "type": "tool_call",
                }
            ],
        ),
        AIMessage(
            content=(
                "APP-1002 numaralı başvuru inceleme aşamasındadır."
            )
        ),
    ]

    with (
        patch(
            "rag_assistant.agent_graph.get_chat_model",
            return_value=model,
        ),
        patch(
            "rag_assistant.agent_graph.create_rag_tool",
            return_value=fake_rag_tool,
        ),
        patch(
            "rag_assistant.agent_graph.create_application_status_tool",
            return_value=fake_application_tool,
        ),
    ):
        graph = build_agent_graph()

        result = graph.invoke({
            "messages": [
                ("user", "APP-1002 başvurusunun durumu nedir?")
            ]
        })

    messages = result["messages"]

    assert len(messages) == 4
    assert messages[1].tool_calls[0]["name"] == "fake_application_tool"
    assert messages[1].tool_calls[0]["args"]["application_id"] == "APP-1002"
    assert messages[2].type == "tool"
    assert "under_review" in messages[2].content
    assert "inceleme aşamasındadır" in messages[-1].content
    assert model_with_tools.invoke.call_count == 2

@tool
def fake_failing_application_tool(application_id: str) -> str:
    """Simulate an application API connection error."""
    return (
        '{"error": "api_request_failed", '
        '"message": "Test connection error"}'
    )

def test_agent_reports_application_api_error():
    model = MagicMock()
    model_with_tools = MagicMock()
    model.bind_tools.return_value = model_with_tools

    model_with_tools.invoke.side_effect = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "fake_failing_application_tool",
                    "args": {"application_id": "APP-1002"},
                    "id": "call_3",
                    "type": "tool_call",
                }
            ],
        ),
        AIMessage(
            content=(
                "Başvuru durumunu şu anda kontrol edemiyorum. "
                "API bağlantısında bir hata oluştu."
            )
        ),
    ]

    with (
        patch(
            "rag_assistant.agent_graph.get_chat_model",
            return_value=model,
        ),
        patch(
            "rag_assistant.agent_graph.create_rag_tool",
            return_value=fake_rag_tool,
        ),
        patch(
            "rag_assistant.agent_graph.create_application_status_tool",
            return_value=fake_failing_application_tool,
        ),
    ):
        graph = build_agent_graph()
        result = graph.invoke({
            "messages": [
                ("user", "APP-1002 başvurusunun durumu nedir?")
            ]
        })

    messages = result["messages"]

    assert messages[1].tool_calls[0]["name"] == (
        "fake_failing_application_tool"
    )
    assert messages[2].type == "tool"
    assert "api_request_failed" in messages[2].content
    assert "kontrol edemiyorum" in messages[-1].content
    assert model_with_tools.invoke.call_count == 2


def test_agent_calls_both_tools_for_combined_question():
    model = MagicMock()
    model_with_tools = MagicMock()
    model.bind_tools.return_value = model_with_tools

    model_with_tools.invoke.side_effect = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "fake_rag_tool",
                    "args": {
                        "question": "Bireysel ihtiyaç kredilerinin faiz oranları nelerdir?"
                    },
                    "id": "call_4",
                    "type": "tool_call",
                },
                {
                    "name": "fake_application_tool",
                    "args": {"application_id": "APP-1002"},
                    "id": "call_5",
                    "type": "tool_call",
                },
            ],
        ),
        AIMessage(
            content=(
                "Bireysel ihtiyaç kredilerinin faiz oranları "
                "%3,73 ile %7,99 arasındadır [S7]. "
                "APP-1002 başvurusu inceleme aşamasındadır."
            )
        ),
    ]

    with (
        patch(
            "rag_assistant.agent_graph.get_chat_model",
            return_value=model,
        ),
        patch(
            "rag_assistant.agent_graph.create_rag_tool",
            return_value=fake_rag_tool,
        ),
        patch(
            "rag_assistant.agent_graph.create_application_status_tool",
            return_value=fake_application_tool,
        ),
    ):
        graph = build_agent_graph()
        result = graph.invoke({
            "messages": [
                (
                    "user",
                    "Bireysel ihtiyaç kredilerinin faiz oranları "
                    "nelerdir? Ayrıca APP-1002 başvurumun durumu nedir?"
                )
            ]
        })

    messages = result["messages"]

    assert len(messages) == 5
    assert len(messages[1].tool_calls) == 2

    called_tools = {
        call["name"] for call in messages[1].tool_calls
    }
    assert called_tools == {
        "fake_rag_tool",
        "fake_application_tool",
    }

    assert messages[2].type == "tool"
    assert messages[3].type == "tool"
    assert "under_review" in messages[2].content or (
        "under_review" in messages[3].content
    )
    assert "[S7]" in messages[-1].content
    assert "inceleme aşamasındadır" in messages[-1].content
    assert model_with_tools.invoke.call_count == 2
