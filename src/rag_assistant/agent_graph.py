
import logging
from time import perf_counter

from langchain_core.messages import SystemMessage
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

from rag_assistant.llm.factory import get_chat_model
from rag_assistant.tools.rag_tool import create_rag_tool
from rag_assistant.tools.application_tool import (
    create_application_status_tool,
)


logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are an enterprise knowledge assistant.

You have access to two tools:

1. search_rag_knowledge:
   Search indexed enterprise documents for information about
   banking products, fees, regulations, and company research.

2. get_application_status:
   Look up a specific loan application using its application ID,
   such as APP-1002. Use this tool for application status,
   approval, and progress questions.

Rules:
1. Select the tool or tools that are relevant to the user's question.
2. Use search_rag_knowledge for questions requiring indexed documents.
3. Use get_application_status for questions about a specific application.
4. Use both tools when the question genuinely requires both sources.
5. Do not invent facts or claim a tool was used when it was not.
6. Base answers on the information returned by the tools.
7. Preserve source references such as [S1] when relevant.
8. Answer in the same language as the user's question.
9. If the available information is insufficient, say so clearly.
10. Do not ask follow-up questions or make suggestions.
"""


def build_agent_graph():
    """Build a tool-calling agent with RAG and application tools."""

    tools = [
        create_rag_tool(),
        create_application_status_tool(),
    ]

    model = get_chat_model().bind_tools(tools)

    def call_model(state: MessagesState) -> dict:
        start_time = perf_counter()
        logger.info("Agent model call started")

        try:
            response = model.invoke([
                SystemMessage(content=SYSTEM_PROMPT),
                *state["messages"],
            ])

            requested_tools = [
                tool_call.get("name", "unknown")
                for tool_call in getattr(response, "tool_calls", [])
            ]

            if requested_tools:
                logger.info(
                    "Model requested tools: %s",
                    ", ".join(requested_tools),
                )
            else:
                logger.info("Model returned a final response")

            logger.info(
                "Agent model call completed in %.3f seconds",
                perf_counter() - start_time,
            )

            return {"messages": [response]}

        except Exception:
            logger.exception("Agent model call failed")
            raise

    builder = StateGraph(MessagesState)

    builder.add_node("agent", call_model)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")

    return builder.compile()
