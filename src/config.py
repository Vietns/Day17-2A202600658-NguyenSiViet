from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from model_provider import ProviderConfig, normalize_provider


@dataclass
class LabConfig:
    """Shared configuration for paths, compact memory, and providers."""

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig
    memory_confidence_threshold: float = 0.75
    memory_decay_days: int = 90


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load `.env` and return a complete lab configuration."""

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    _load_dotenv(root / ".env")

    provider = normalize_provider(os.getenv("LLM_PROVIDER", "mock"))
    judge_provider = normalize_provider(os.getenv("JUDGE_PROVIDER", provider))
    state_dir = Path(os.getenv("STATE_DIR", str(root / "state"))).resolve()
    state_dir.mkdir(parents=True, exist_ok=True)

    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=state_dir,
        compact_threshold_tokens=int(os.getenv("COMPACT_THRESHOLD_TOKENS", "1200")),
        compact_keep_messages=int(os.getenv("COMPACT_KEEP_MESSAGES", "8")),
        model=_provider_config(provider, "LLM_MODEL"),
        judge_model=_provider_config(judge_provider, "JUDGE_MODEL"),
        memory_confidence_threshold=float(os.getenv("MEMORY_CONFIDENCE_THRESHOLD", "0.75")),
        memory_decay_days=int(os.getenv("MEMORY_DECAY_DAYS", "90")),
    )


def _load_dotenv(path: Path) -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(path)
        return
    except Exception:
        pass

    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _provider_config(provider: str, model_env: str) -> ProviderConfig:
    defaults = {
        "mock": "offline-mock",
        "openai": "gpt-4o-mini",
        "custom": "gpt-4o-mini",
        "gemini": "gemini-1.5-flash",
        "anthropic": "claude-3-5-haiku-latest",
        "ollama": "llama3.1",
        "openrouter": "openai/gpt-4o-mini",
    }
    api_keys = {
        "openai": os.getenv("OPENAI_API_KEY"),
        "custom": os.getenv("CUSTOM_API_KEY"),
        "gemini": os.getenv("GEMINI_API_KEY"),
        "anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "openrouter": os.getenv("OPENROUTER_API_KEY"),
    }
    base_urls = {
        "custom": os.getenv("CUSTOM_BASE_URL"),
        "ollama": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        "openrouter": os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    }
    return ProviderConfig(
        provider=provider,
        model_name=os.getenv(model_env, defaults.get(provider, "offline-mock")),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
        api_key=api_keys.get(provider),
        base_url=base_urls.get(provider),
    )
