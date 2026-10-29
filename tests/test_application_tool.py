import json
import requests

from rag_assistant.tools.application_tool import (
    create_application_status_tool,
)

def test_application_tool_handles_connection_error(monkeypatch):
    def raise_connection_error(*args, **kwargs):
        raise requests.ConnectionError("Test connection error")

    monkeypatch.setattr(
        "rag_assistant.tools.application_tool.requests.get",
        raise_connection_error,
    )

    tool = create_application_status_tool()
    result = json.loads(
        tool.invoke({"application_id": "APP-1002"})
    )

    assert result["error"] == "api_request_failed"
    assert "Test connection error" in result["message"]

def test_application_tool_handles_missing_application(monkeypatch):
    def return_404(*args, **kwargs):
        response = requests.Response()
        response.status_code = 404
        return response

    monkeypatch.setattr(
        "rag_assistant.tools.application_tool.requests.get",
        return_404,
    )

    tool = create_application_status_tool()
    result = json.loads(
        tool.invoke({"application_id": "APP-9999"})
    )

    assert result["error"] == "application_not_found"
    assert result["application_id"] == "APP-9999"
