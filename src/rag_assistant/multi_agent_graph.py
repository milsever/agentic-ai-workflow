import json
import logging
from time import perf_counter
from typing import TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph, MessagesState

from rag_assistant.llm.factory import get_chat_model
from rag_assistant.tools.rag_tool import create_rag_tool
from rag_assistant.tools.application_tool import (
    create_application_status_tool,
)


logger = logging.getLogger(__name__)


class MultiAgentState(MessagesState, total=False):
    routes: list[str]
    rag_question: str
    application_id: str
    rag_result: str
    application_result: str
    router_response: str


@tool
def route_to_rag(question: str) -> str:
    """Route an enterprise-document question to the RAG specialist."""
    return "RAG specialist selected."


@tool
def route_to_application(application_id: str) -> str:
    """Route a specific loan application query to the application specialist."""
    return "Application specialist selected."


ORCHESTRATOR_PROMPT = """You are the orchestrator of an enterprise assistant.

Available specialists:
1. route_to_rag: for banking products, fees, regulations, and company research
   that require indexed enterprise documents. Pass the user's relevant question.
2. route_to_application: for the status or progress of a specific loan
   application. Pass its application ID, such as APP-1002.

Rules:
- Select every specialist genuinely needed to answer the user.
- Select both specialists when the request needs both sources.
- Do not select a specialist if it is irrelevant.
- Do not invent application IDs.
- If the request is unrelated to both specialists, respond briefly that
  this workflow supports enterprise-document questions and loan application
  status queries.
"""


FINALIZER_PROMPT = """You are an enterprise assistant combining specialist results.

Rules:
- Answer in the same language as the user's question.
- Use only the supplied specialist results.
- Preserve RAG source references such as [S1].
- Clearly communicate tool/API errors; never invent missing information.
- If only one specialist was needed, answer using that result.
- For combined requests, address both parts in one coherent answer.
"""


def build_multi_agent_graph():
    """Build an orchestrator with dedicated RAG and application specialists."""

    rag_tool = create_rag_tool()
    application_tool = create_application_status_tool()

    model = get_chat_model()
    router_model = model.bind_tools([
        route_to_rag,
        route_to_application,
    ])

    def orchestrator(state: MultiAgentState) -> dict:
        start_time = perf_counter()
        logger.info("Multi-agent orchestration started")

        try:
            response = router_model.invoke([
                SystemMessage(content=ORCHESTRATOR_PROMPT),
                *state["messages"],
            ])

            routes = []
            rag_question = ""
            application_id = ""

            for tool_call in getattr(response, "tool_calls", []):
                name = tool_call.get("name")
                args = tool_call.get("args", {})

                if name == "route_to_rag":
                    question = args.get("question", "").strip()
                    if question:
                        routes.append("rag_agent")
                        rag_question = question

                elif name == "route_to_application":
                    app_id = args.get("application_id", "").strip()
                    if app_id:
                        routes.append("application_agent")
                        application_id = app_id

            # Keep the execution order deterministic and avoid duplicate routes.
            routes = list(dict.fromkeys(routes))

            logger.info(
                "Orchestrator selected specialists: %s",
                routes or "none",
            )
            logger.info(
                "Orchestration completed in %.3f seconds",
                perf_counter() - start_time,
            )

            return {
                "routes": routes,
                "rag_question": rag_question,
                "application_id": application_id,
                "router_response": response.content or "",
            }

        except Exception:
            logger.exception("Multi-agent orchestration failed")
            raise

    def route_request(state: MultiAgentState) -> str:
        routes = state.get("routes", [])

        if "rag_agent" in routes:
            return "rag_agent"

        if "application_agent" in routes:
            return "application_agent"

        return "finalize"

    def rag_agent(state: MultiAgentState) -> dict:
        logger.info("RAG specialist started")

        try:
            result = rag_tool.invoke({
                "question": state["rag_question"],
            })
        except Exception as exc:
            logger.exception("RAG specialist failed")
            result = json.dumps({
                "error": "rag_specialist_failed",
                "message": str(exc),
            }, ensure_ascii=False)

        return {"rag_result": result}

    def application_agent(state: MultiAgentState) -> dict:
        logger.info("Application specialist started")

        try:
            result = application_tool.invoke({
                "application_id": state["application_id"],
            })
        except Exception as exc:
            logger.exception("Application specialist failed")
            result = json.dumps({
                "error": "application_specialist_failed",
                "message": str(exc),
            }, ensure_ascii=False)

        return {"application_result": result}

    def after_rag(state: MultiAgentState) -> str:
        if "application_agent" in state.get("routes", []):
            return "application_agent"
        return "finalize"

    def finalize(state: MultiAgentState) -> dict:
        rag_result = state.get("rag_result", "")
        application_result = state.get("application_result", "")

        # If neither specialist was selected, return the orchestrator's
        # unsupported-request response without making another model call.
        if not rag_result and not application_result:
            answer = state.get("router_response", "")
            if not answer:
                answer = (
                    "Bu iş akışı kurumsal doküman soruları ve kredi "
                    "başvurusu durumu sorgularını destekliyor."
                )
            return {"messages": [AIMessage(content=answer)]}

        user_request = next(
            (
                message.content
                for message in reversed(state["messages"])
                if isinstance(message, HumanMessage)
            ),
            "",
        )

        prompt = (
            f"User request:\n{user_request}\n\n"
            f"RAG specialist result:\n{rag_result or 'Not requested.'}\n\n"
            f"Application specialist result:\n"
            f"{application_result or 'Not requested.'}"
        )

        start_time = perf_counter()
        logger.info("Final response synthesis started")

        try:
            response = model.invoke([
                SystemMessage(content=FINALIZER_PROMPT),
                HumanMessage(content=prompt),
            ])

            logger.info(
                "Final response synthesis completed in %.3f seconds",
                perf_counter() - start_time,
            )
            return {"messages": [response]}

        except Exception:
            logger.exception("Final response synthesis failed")
            raise

    builder = StateGraph(MultiAgentState)

    builder.add_node("orchestrator", orchestrator)
    builder.add_node("rag_agent", rag_agent)
    builder.add_node("application_agent", application_agent)
    builder.add_node("finalize", finalize)

    builder.add_edge(START, "orchestrator")

    builder.add_conditional_edges(
        "orchestrator",
        route_request,
        {
            "rag_agent": "rag_agent",
            "application_agent": "application_agent",
            "finalize": "finalize",
        },
    )

    builder.add_conditional_edges(
        "rag_agent",
        after_rag,
        {
            "application_agent": "application_agent",
            "finalize": "finalize",
        },
    )

    builder.add_edge("application_agent", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile()