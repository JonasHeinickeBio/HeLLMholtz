from dataclasses import dataclass, field
import os

from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()


@dataclass
class Settings:
    """Centralized configuration for Helmholtz LLM Suite."""

    default_models: list[str] = field(default_factory=list)
    blablador_api_key: str | None = None
    blablador_base_url: str | None = None
    timeout_seconds: float = 30.0

    # Provider keys (read directly from env, but can be accessed here if needed)
    openai_api_key: str | None = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    anthropic_api_key: str | None = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))
    google_api_key: str | None = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY"))

    # AnythingLLM integration
    anythingllm_base_url: str | None = field(
        default_factory=lambda: os.getenv("ANYTHINGLLM_BASE_URL", "http://localhost:3001")
    )
    anythingllm_api_key: str | None = field(
        default_factory=lambda: os.getenv("ANYTHINGLLM_API_KEY")
    )


def get_settings() -> Settings:
    """Load settings from environment variables."""

    # Parse default models
    models_str = os.getenv("AISUITE_DEFAULT_MODELS", "")
    default_models = [m.strip() for m in models_str.split(",") if m.strip()]

    # Blablador config
    blablador_key = os.getenv("BLABLADOR_API_KEY")
    blablador_url = os.getenv("BLABLADOR_API_BASE")

    # Timeout
    try:
        timeout = float(os.getenv("HELMHOLTZ_TIMEOUT_SECONDS", "30.0"))
    except ValueError:
        timeout = 30.0

    return Settings(
        default_models=default_models,
        blablador_api_key=blablador_key,
        blablador_base_url=blablador_url,
        timeout_seconds=timeout,
    )
