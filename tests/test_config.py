import os
from unittest.mock import patch
from hellmholtz.core.config import get_settings

def test_get_settings_defaults() -> None:
    with patch.dict(os.environ, {}, clear=True):
        settings = get_settings()
        assert settings.default_models == []
        assert settings.timeout_seconds == 30.0
        assert settings.blablador_api_key is None

def test_settings_env_override(monkeypatch: object) -> None:
    env_vars = {
        "AISUITE_DEFAULT_MODELS": "openai:gpt-4o, ollama:llama3",
        "BLABLADOR_API_KEY": "test-key",
        "HELMHOLTZ_TIMEOUT_SECONDS": "60.0"
    }
    with patch.dict(os.environ, env_vars, clear=True):
        settings = get_settings()
        assert settings.default_models == ["openai:gpt-4o", "ollama:llama3"]
        assert settings.blablador_api_key == "test-key"
        assert settings.timeout_seconds == 60.0
