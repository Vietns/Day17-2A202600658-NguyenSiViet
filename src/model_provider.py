from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProviderConfig:
    """Configuration shared by every supported chat provider.

    If a provider package or API key is missing, callers can safely treat
    ``build_chat_model`` returning ``None`` as offline/mock mode.
    """

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    """Normalize provider names and common aliases/misspellings."""

    aliases = {
        "": "mock",
        "offline": "mock",
        "none": "mock",
        "mock": "mock",
        "open_ai": "openai",
        "openai-compatible": "custom",
        "openai_compatible": "custom",
        "google": "gemini",
        "google-genai": "gemini",
        "anthorpic": "anthropic",
        "claude": "anthropic",
        "local": "ollama",
    }
    key = (value or "mock").strip().lower().replace(" ", "-")
    return aliases.get(key, key)


def build_chat_model(config: ProviderConfig):
    """Instantiate a LangChain chat model when dependencies and credentials exist.

    The lab must run without API keys, so every failure returns ``None`` and the
    agents use their deterministic offline path.
    """

    provider = normalize_provider(config.provider)
    if provider == "mock":
        return None

    try:
        if provider in {"openai", "custom"}:
            if provider == "openai" and not config.api_key:
                return None
            if provider == "custom" and not (config.api_key and config.base_url):
                return None
            from langchain_openai import ChatOpenAI

            kwargs = {
                "model": config.model_name,
                "temperature": config.temperature,
                "api_key": config.api_key,
            }
            if config.base_url:
                kwargs["base_url"] = config.base_url
            return ChatOpenAI(**kwargs)

        if provider == "gemini":
            if not config.api_key:
                return None
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(
                model=config.model_name,
                temperature=config.temperature,
                google_api_key=config.api_key,
            )

        if provider == "anthropic":
            if not config.api_key:
                return None
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(
                model=config.model_name,
                temperature=config.temperature,
                api_key=config.api_key,
            )

        if provider == "ollama":
            from langchain_ollama import ChatOllama

            kwargs = {"model": config.model_name, "temperature": config.temperature}
            if config.base_url:
                kwargs["base_url"] = config.base_url
            return ChatOllama(**kwargs)

        if provider == "openrouter":
            if not config.api_key:
                return None
            try:
                from langchain_openrouter import ChatOpenRouter

                return ChatOpenRouter(
                    model=config.model_name,
                    temperature=config.temperature,
                    api_key=config.api_key,
                )
            except ImportError:
                from langchain_openai import ChatOpenAI

                return ChatOpenAI(
                    model=config.model_name,
                    temperature=config.temperature,
                    api_key=config.api_key,
                    base_url=config.base_url or "https://openrouter.ai/api/v1",
                )
    except Exception:
        return None

    return None
