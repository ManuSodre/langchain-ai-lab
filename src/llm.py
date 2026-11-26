from langchain_openai import ChatOpenAI

from src.config import OPENAI_MODEL, get_openai_api_key


def get_chat_model() -> ChatOpenAI:
    """Create the configured OpenAI chat model."""
    return ChatOpenAI(
        model=OPENAI_MODEL,
        api_key=get_openai_api_key(),
        temperature=0,
    )
