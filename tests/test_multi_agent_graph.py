from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from rag_assistant.multi_agent_graph import build_multi_agent_graph


@tool
def fake_rag_tool(question: str) -> str:
    """Return a test answer from enterprise documents."""
    return (
        '{"answer": "Transfer fee is 5 TL [S1].", '
        '"sources": [{"source": "fees.pdf", "page": 2}]}'
    )


@tool
def fake_application_tool(application_id: str) -> str:
    """Return a test application status."""
    return (
        '{"application_id": "APP-1002", '
        '"status": "under_review"}'
    )


def make_model():
    model = MagicMock()
    router_model = MagicMock()
    model.bind_tools.return_value = router_model
    return model, router_model


def test_multi_agent_routes_to_rag_specialist():
    model, router_model = make_model()

    router_model.invoke.return_value = AIMessage(
        content="",
        tool_calls=[{
            "name": "route_to_rag",
            "args": {"question": "What is the transfer fee?"},
            "id": "route_1",
            "type": "tool_call",
        }],
    )
    model.invoke.return_value = AIMessage(
        content="The transfer fee is 5 TL [S1]."
    )

    with (
        patch(
            "rag_assistant.multi_agent_graph.get_chat_model",
            return_value=model,
        ),
        patch(
            "rag_assistant.multi_agent_graph.create_rag_tool",
            return_value=fake_rag_tool,
        ),
        patch(
            "rag_assistant.multi_agent_graph.create_application_status_tool",
            return_value=fake_application_tool,
        ),
    ):
        graph = build_multi_agent_graph()
        result = graph.invoke({
            "messages": [("user", "What is the transfer fee?")]
        })

    assert result["messages"][-1].content == (
        "The transfer fee is 5 TL [S1]."
    )
    assert model.invoke.call_count == 1
    assert router_model.invoke.call_count == 1


def test_multi_agent_routes_to_application_specialist():
    model, router_model = make_model()

    router_model.invoke.return_value = AIMessage(
        content="",
        tool_calls=[{
            "name": "route_to_application",
            "args": {"application_id": "APP-1002"},
            "id": "route_2",
            "type": "tool_call",
        }],
    )
    model.invoke.return_value = AIMessage(
        content="APP-1002 başvurusu inceleme aşamasındadır."
    )

    with (
        patch(
            "rag_assistant.multi_agent_graph.get_chat_model",
            return_value=model,
        ),
        patch(
            "rag_assistant.multi_agent_graph.create_rag_tool",
            return_value=fake_rag_tool,
        ),
        patch(
            "rag_assistant.multi_agent_graph.create_application_status_tool",
            return_value=fake_application_tool,
        ),
    ):
        graph = build_multi_agent_graph()
        result = graph.invoke({
            "messages": [
                ("user", "APP-1002 başvurumun durumu nedir?")
            ]
        })

    assert "inceleme aşamasındadır" in result["messages"][-1].content
    model.invoke.assert_called_once()


def test_multi_agent_runs_both_specialists_for_combined_question():
    model, router_model = make_model()

    router_model.invoke.return_value = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "route_to_rag",
                "args": {
                    "question": "Bireysel ihtiyaç kredilerinin faiz oranları nelerdir?"
                },
                "id": "route_3",
                "type": "tool_call",
            },
            {
                "name": "route_to_application",
                "args": {"application_id": "APP-1002"},
                "id": "route_4",
                "type": "tool_call",
            },
        ],
    )
    model.invoke.return_value = AIMessage(
        content=(
            "Kredi bilgileri [S1] kaynaklıdır. "
            "APP-1002 başvurusu inceleme aşamasındadır."
        )
    )

    with (
        patch(
            "rag_assistant.multi_agent_graph.get_chat_model",
            return_value=model,
        ),
        patch(
            "rag_assistant.multi_agent_graph.create_rag_tool",
            return_value=fake_rag_tool,
        ) as rag_factory,
        patch(
            "rag_assistant.multi_agent_graph.create_application_status_tool",
            return_value=fake_application_tool,
        ) as app_factory,
    ):
        graph = build_multi_agent_graph()
        result = graph.invoke({
            "messages": [
                (
                    "user",
                    "Kredi faizleri nedir ve APP-1002 başvurumun durumu nedir?",
                )
            ]
        })

    answer = result["messages"][-1].content

    assert "[S1]" in answer
    assert "APP-1002" in answer
    rag_factory.assert_called_once()
    app_factory.assert_called_once()
    model.invoke.assert_called_once()


def test_multi_agent_handles_request_with_no_matching_specialist():
    model, router_model = make_model()

    router_model.invoke.return_value = AIMessage(
        content="Bu iş akışı yalnızca belirli kurumsal soruları destekler."
    )

    with (
        patch(
            "rag_assistant.multi_agent_graph.get_chat_model",
            return_value=model,
        ),
        patch(
            "rag_assistant.multi_agent_graph.create_rag_tool",
            return_value=fake_rag_tool,
        ),
        patch(
            "rag_assistant.multi_agent_graph.create_application_status_tool",
            return_value=fake_application_tool,
        ),
    ):
        graph = build_multi_agent_graph()
        result = graph.invoke({
            "messages": [("user", "Bana bir şiir yazar mısın?")]
        })

    assert result["messages"][-1].content == (
        "Bu iş akışı yalnızca belirli kurumsal soruları destekler."
    )
    model.invoke.assert_not_called()
