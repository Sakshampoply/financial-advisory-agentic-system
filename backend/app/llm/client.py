from langchain_openai import ChatOpenAI

from app.config import settings
from app.observability.langfuse_setup import get_langfuse_callbacks

_HEADERS = {
    "HTTP-Referer": "http://localhost:3000",
    "X-Title": "Financial Advisory System",
}


def get_chat_model(streaming: bool = False, temperature: float = 0.1) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.OPENROUTER_MODEL,
        openai_api_key=settings.OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
        streaming=streaming,
        temperature=temperature,
        request_timeout=60,
        default_headers=_HEADERS,
        callbacks=get_langfuse_callbacks(),
    )
