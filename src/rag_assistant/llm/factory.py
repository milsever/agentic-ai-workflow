
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI, AzureChatOpenAI

from rag_assistant import config

def get_chat_model() -> BaseChatModel:
    """Create a chat model based on the selected provider."""

    provider = config.LLM_PROVIDER

    if provider == "ollama":
        return ChatOllama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0,
        )

    if provider == "openai":
        if not config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is missing in .env")

        return ChatOpenAI(
            model=config.OPENAI_MODEL,
            api_key=config.OPENAI_API_KEY,
            temperature=0,
        )

    if provider == "azure":
        required = {
            "AZURE_OPENAI_API_KEY": config.AZURE_OPENAI_API_KEY,
            "AZURE_OPENAI_ENDPOINT": config.AZURE_OPENAI_ENDPOINT,
            "AZURE_OPENAI_DEPLOYMENT": config.AZURE_OPENAI_DEPLOYMENT,
            "AZURE_OPENAI_API_VERSION": config.AZURE_OPENAI_API_VERSION,
        }
        missing = [key for key, value in required.items() if not value]

        if missing:
            raise ValueError(
                f"Missing Azure configuration: {', '.join(missing)}"
            )

        return AzureChatOpenAI(
            azure_deployment=config.AZURE_OPENAI_DEPLOYMENT,
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
            api_key=config.AZURE_OPENAI_API_KEY,
            api_version=config.AZURE_OPENAI_API_VERSION,
            temperature=0,
        )

    raise ValueError(f"Unsupported LLM provider: {provider}")
