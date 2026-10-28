# Project instructions

- Keep application code under `src/rag_assistant/` and tests under `tests/`.
- Keep credentials in environment variables; never commit `.env` or secrets.
- Store source documents under `data/documents/` and generated vector data under `data/chroma/`.
- Use LangGraph for multi-step stateful workflows.
- Consult the appropriate LangChain MCP documentation before implementing or answering questions about LangChain, LangGraph, LangSmith, their agents, middleware, MCP, or SDK/API usage.
- Keep LLM provider configuration centralized in the factory module.
- Write tests for core functionality.
- Keep document ingestion, retrieval, generation, and API layers modular.

