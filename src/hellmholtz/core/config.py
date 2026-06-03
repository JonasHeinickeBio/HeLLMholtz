from dataclasses import dataclass, field
import os
from pathlib import Path

from dotenv import load_dotenv

# Configuration directories
USER_CONFIG_DIR = Path.home() / ".config" / "hellmholtz"
USER_CONFIG_FILE = USER_CONFIG_DIR / ".env"


def _load_user_config() -> None:
    """Load user-level configuration if available."""
    # First, try to load from user config directory
    if USER_CONFIG_FILE.exists():
        load_dotenv(USER_CONFIG_FILE, override=False)

    # Then load project-local .env with override=True so it replaces user config
    load_dotenv(override=True)


# Load configuration with user-level support
_load_user_config()


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
