"""Tests for configuration exporters."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from hellmholtz.core.exporters import (
    AiderExporter,
    ClaudeCodeExporter,
    ContinueExporter,
    CursorExporter,
    EXPORTERS,
    GPT4AllExporter,
    GenericOpenAIExporter,
    get_exporter,
    HermesAgentExporter,
    JanAIExporter,
    LangChainExporter,
    list_exporters,
    OpenCodeExporter,
    PiAgentExporter,
)
from hellmholtz.core.model_manager import ModelConfig


@pytest.fixture
def sample_model() -> ModelConfig:
    """Create a sample model for testing."""
    return ModelConfig(
        name="Test Model",
        provider="blablador",
        model="test-model-1",
        api_base="https://api.example.com/v1",
        api_key="test-api-key-123",
        context_length=8192,
        max_tokens=4096,
        roles=["chat"],
    )


@pytest.fixture
def sample_models(sample_model: ModelConfig) -> list[ModelConfig]:
    """Create a list of sample models for testing."""
    return [
        sample_model,
        ModelConfig(
            name="Second Model",
            provider="blablador",
            model="test-model-2",
            api_base="https://api.example.com/v1",
            api_key="test-api-key-123",
            context_length=4096,
            max_tokens=2048,
            roles=["chat", "completion"],
        ),
    ]


class TestConfigExporterBase:
    """Tests for the base ConfigExporter class."""

    def test_load_existing_missing_file(self, tmp_path: Path) -> None:
        """Test _load_existing with a non-existent file."""
        exporter = OpenCodeExporter()
        result = exporter._load_existing(tmp_path / "nonexistent.json")
        assert result is None

    def test_load_existing_json_file(self, tmp_path: Path) -> None:
        """Test _load_existing with a valid JSON file."""
        exporter = OpenCodeExporter()
        json_file = tmp_path / "config.json"
        json_file.write_text('{"key": "value", "number": 42}')

        result = exporter._load_existing(json_file)
        assert result == {"key": "value", "number": 42}

    def test_load_existing_yaml_file(self, tmp_path: Path) -> None:
        """Test _load_existing with a valid YAML file."""
        exporter = OpenCodeExporter()
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("key: value\nnumber: 42\n")

        result = exporter._load_existing(yaml_file)
        assert result == {"key": "value", "number": 42}

    def test_load_existing_yml_file(self, tmp_path: Path) -> None:
        """Test _load_existing with a .yml file."""
        exporter = OpenCodeExporter()
        yml_file = tmp_path / "config.yml"
        yml_file.write_text("key: value\nlist:\n  - item1\n  - item2\n")

        result = exporter._load_existing(yml_file)
        assert result == {"key": "value", "list": ["item1", "item2"]}

    def test_load_existing_corrupt_json(self, tmp_path: Path) -> None:
        """Test _load_existing with corrupt JSON."""
        exporter = OpenCodeExporter()
        json_file = tmp_path / "corrupt.json"
        json_file.write_text("{invalid json content")

        result = exporter._load_existing(json_file)
        assert result is None

    def test_load_existing_corrupt_yaml(self, tmp_path: Path) -> None:
        """Test _load_existing with corrupt YAML."""
        exporter = OpenCodeExporter()
        yaml_file = tmp_path / "corrupt.yaml"
        yaml_file.write_text(":\n  invalid:\n    - : [")

        result = exporter._load_existing(yaml_file)
        assert result is None

    def test_load_existing_unknown_extension(self, tmp_path: Path) -> None:
        """Test _load_existing with unknown file extension."""
        exporter = OpenCodeExporter()
        txt_file = tmp_path / "config.txt"
        txt_file.write_text("some content")

        result = exporter._load_existing(txt_file)
        assert result is None


class TestOpenCodeExporter:
    """Tests for OpenCode exporter."""

    def test_tool_name(self) -> None:
        """Test tool_name property."""
        exporter = OpenCodeExporter()
        assert exporter.tool_name == "opencode"

    def test_config_path(self) -> None:
        """Test config_path property."""
        exporter = OpenCodeExporter()
        with patch.object(Path, "home", return_value=Path("/mock/home")):
            path = exporter.config_path
            assert path == Path("/mock/home/.config/opencode/opencode.json")

    def test_export_creates_file(self, tmp_path: Path, sample_model: ModelConfig) -> None:
        """Test export creates a new config file."""
        exporter = OpenCodeExporter()
        output_file = tmp_path / "opencode.json"

        result = exporter.export([sample_model], output_path=output_file)

        assert result == output_file
        assert output_file.exists()

        config = json.loads(output_file.read_text())
        assert "provider" in config
        assert "blablador" in config["provider"]
        assert "models" in config["provider"]["blablador"]
        assert "test-model-1" in config["provider"]["blablador"]["models"]

    def test_export_with_context_length(
        self, tmp_path: Path, sample_model: ModelConfig
    ) -> None:
        """Test export with context length set."""
        exporter = OpenCodeExporter()
        output_file = tmp_path / "opencode.json"

        exporter.export([sample_model], output_path=output_file)

        config = json.loads(output_file.read_text())
        model_config = config["provider"]["blablador"]["models"]["test-model-1"]
        assert model_config["limit"]["context"] == 8192
        assert model_config["limit"]["output"] == 4096

    def test_export_without_context_length(self, tmp_path: Path) -> None:
        """Test export with no context length uses defaults."""
        exporter = OpenCodeExporter()
        output_file = tmp_path / "opencode.json"
        model = ModelConfig(name="Test", model="m1", api_base="http://x", api_key="k")

        exporter.export([model], output_path=output_file)

        config = json.loads(output_file.read_text())
        model_config = config["provider"]["blablador"]["models"]["m1"]
        assert model_config["limit"]["context"] == 98304
        assert model_config["limit"]["output"] == 32768

    def test_export_empty_models(self, tmp_path: Path) -> None:
        """Test export with empty models list."""
        exporter = OpenCodeExporter()
        output_file = tmp_path / "opencode.json"

        result = exporter.export([], output_path=output_file)

        assert result == output_file
        config = json.loads(output_file.read_text())
        assert config["provider"]["blablador"]["options"]["baseURL"] == ""
        assert config["provider"]["blablador"]["options"]["apiKey"] == ""
        assert config["provider"]["blablador"]["models"] == {}

    def test_export_merge_existing(self, tmp_path: Path) -> None:
        """Test export merges with existing config."""
        exporter = OpenCodeExporter()
        output_file = tmp_path / "opencode.json"

        existing_config = {
            "provider": {
                "blablador": {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "Blablador",
                    "options": {"baseURL": "old-url", "apiKey": "old-key"},
                    "models": {"old-model": {"name": "Old Model"}},
                }
            },
            "other_key": "should_be_preserved",
        }
        output_file.write_text(json.dumps(existing_config))

        model = ModelConfig(
            name="New Model", model="new-m", api_base="new-url", api_key="new-key"
        )
        exporter.export([model], output_path=output_file, merge=True)

        config = json.loads(output_file.read_text())
        assert config["other_key"] == "should_be_preserved"
        assert config["provider"]["blablador"]["options"]["apiKey"] == "new-key"
        assert "new-m" in config["provider"]["blablador"]["models"]

    def test_export_no_merge(self, tmp_path: Path, sample_model: ModelConfig) -> None:
        """Test export without merge overwrites existing."""
        exporter = OpenCodeExporter()
        output_file = tmp_path / "opencode.json"

        output_file.write_text('{"old": "config"}')
        exporter.export([sample_model], output_path=output_file, merge=False)

        config = json.loads(output_file.read_text())
        assert "old" not in config
        assert "provider" in config

    def test_export_multiple_models(
        self, tmp_path: Path, sample_models: list[ModelConfig]
    ) -> None:
        """Test export with multiple models."""
        exporter = OpenCodeExporter()
        output_file = tmp_path / "opencode.json"

        exporter.export(sample_models, output_path=output_file)

        config = json.loads(output_file.read_text())
        models = config["provider"]["blablador"]["models"]
        assert len(models) == 2
        assert "test-model-1" in models
        assert "test-model-2" in models


class TestClaudeCodeExporter:
    """Tests for Claude Code exporter."""

    def test_tool_name(self) -> None:
        """Test tool_name property."""
        exporter = ClaudeCodeExporter()
        assert exporter.tool_name == "claude-code"

    def test_config_path(self) -> None:
        """Test config_path property."""
        exporter = ClaudeCodeExporter()
        with patch.object(Path, "home", return_value=Path("/mock/home")):
            path = exporter.config_path
            assert path == Path("/mock/home/.claude/settings.json")

    def test_export_creates_file(self, tmp_path: Path, sample_model: ModelConfig) -> None:
        """Test export creates settings.json."""
        exporter = ClaudeCodeExporter()
        output_file = tmp_path / "settings.json"

        result = exporter.export([sample_model], output_path=output_file)

        assert result == output_file
        config = json.loads(output_file.read_text())
        assert config["env"]["ANTHROPIC_BASE_URL"] == "https://api.example.com/v1"
        assert config["env"]["ANTHROPIC_API_KEY"] == "test-api-key-123"
        assert config["model"] == "test-model-1"

    def test_export_single_model_sets_model_key(self, tmp_path: Path) -> None:
        """Test export with single model sets model key."""
        exporter = ClaudeCodeExporter()
        output_file = tmp_path / "settings.json"
        model = ModelConfig(name="M", model="my-model", api_base="b", api_key="k")

        exporter.export([model], output_path=output_file)

        config = json.loads(output_file.read_text())
        assert config["model"] == "my-model"

    def test_export_multiple_models_no_model_key(
        self, tmp_path: Path, sample_models: list[ModelConfig]
    ) -> None:
        """Test export with multiple models doesn't set model key."""
        exporter = ClaudeCodeExporter()
        output_file = tmp_path / "settings.json"

        exporter.export(sample_models, output_path=output_file)

        config = json.loads(output_file.read_text())
        assert "model" not in config
        assert "ANTHROPIC_BASE_URL" in config["env"]

    def test_export_empty_models(self, tmp_path: Path) -> None:
        """Test export with empty models list."""
        exporter = ClaudeCodeExporter()
        output_file = tmp_path / "settings.json"

        exporter.export([], output_path=output_file)

        config = json.loads(output_file.read_text())
        assert config == {"env": {}}

    def test_export_merge_existing(self, tmp_path: Path) -> None:
        """Test export merges with existing config."""
        exporter = ClaudeCodeExporter()
        output_file = tmp_path / "settings.json"

        existing = {"env": {"SOME_VAR": "value"}, "existing_key": "preserved"}
        output_file.write_text(json.dumps(existing))

        model = ModelConfig(name="M", model="m", api_base="b", api_key="k")
        exporter.export([model], output_path=output_file, merge=True)

        config = json.loads(output_file.read_text())
        assert config["existing_key"] == "preserved"
        assert config["env"]["SOME_VAR"] == "value"
        assert config["env"]["ANTHROPIC_BASE_URL"] == "b"


class TestContinueExporter:
    """Tests for Continue.dev exporter."""

    def test_tool_name(self) -> None:
        """Test tool_name property."""
        exporter = ContinueExporter()
        assert exporter.tool_name == "continue"

    def test_config_path(self) -> None:
        """Test config_path property."""
        exporter = ContinueExporter()
        with patch.object(Path, "home", return_value=Path("/mock/home")):
            path = exporter.config_path
            assert path == Path("/mock/home/.continue/config.yaml")

    def test_export_creates_yaml(
        self, tmp_path: Path, sample_model: ModelConfig
    ) -> None:
        """Test export creates YAML config."""
        exporter = ContinueExporter()
        output_file = tmp_path / "config.yaml"

        result = exporter.export([sample_model], output_path=output_file)

        assert result == output_file
        config = yaml.safe_load(output_file.read_text())
        assert config["name"] == "Blablador Configuration"
        assert config["version"] == "1.0.0"
        assert config["schema"] == "v1"
        assert len(config["models"]) == 1
        assert config["models"][0]["name"] == "Test Model"

    def test_export_model_with_context_length(
        self, tmp_path: Path, sample_model: ModelConfig
    ) -> None:
        """Test export with context length includes completion options."""
        exporter = ContinueExporter()
        output_file = tmp_path / "config.yaml"

        exporter.export([sample_model], output_path=output_file)

        config = yaml.safe_load(output_file.read_text())
        model = config["models"][0]
        assert "defaultCompletionOptions" in model
        assert model["defaultCompletionOptions"]["contextLength"] == 8192
        assert model["defaultCompletionOptions"]["maxTokens"] == 4096

    def test_export_model_without_context_length(self, tmp_path: Path) -> None:
        """Test export without context length skips completion options."""
        exporter = ContinueExporter()
        output_file = tmp_path / "config.yaml"
        model = ModelConfig(name="M", model="m", api_base="b", api_key="k")

        exporter.export([model], output_path=output_file)

        config = yaml.safe_load(output_file.read_text())
        assert "defaultCompletionOptions" not in config["models"][0]

    def test_export_model_roles(
        self, tmp_path: Path, sample_model: ModelConfig
    ) -> None:
        """Test export includes model roles."""
        exporter = ContinueExporter()
        output_file = tmp_path / "config.yaml"

        exporter.export([sample_model], output_path=output_file)

        config = yaml.safe_load(output_file.read_text())
        assert config["models"][0]["roles"] == ["chat"]

    def test_export_empty_models(self, tmp_path: Path) -> None:
        """Test export with empty models list."""
        exporter = ContinueExporter()
        output_file = tmp_path / "config.yaml"

        exporter.export([], output_path=output_file)

        config = yaml.safe_load(output_file.read_text())
        assert config["models"] == []
        assert config["name"] == "Blablador Configuration"

    def test_export_merge_existing(self, tmp_path: Path) -> None:
        """Test export merges with existing YAML config."""
        exporter = ContinueExporter()
        output_file = tmp_path / "config.yaml"

        existing = {"name": "Custom Name", "custom_key": "value"}
        output_file.write_text(yaml.dump(existing))

        model = ModelConfig(name="M", model="m", api_base="b", api_key="k")
        exporter.export([model], output_path=output_file, merge=True)

        config = yaml.safe_load(output_file.read_text())
        assert config["custom_key"] == "value"
        assert config["name"] == "Custom Name"
        assert len(config["models"]) == 1


class TestAiderExporter:
    """Tests for Aider exporter."""

    def test_tool_name(self) -> None:
        """Test tool_name property."""
        exporter = AiderExporter()
        assert exporter.tool_name == "aider"

    def test_config_path(self) -> None:
        """Test config_path property."""
        exporter = AiderExporter()
        with patch.object(Path, "home", return_value=Path("/mock/home")):
            path = exporter.config_path
            assert path == Path("/mock/home/.aider.conf.yml")

    def test_export_creates_yaml(
        self, tmp_path: Path, sample_model: ModelConfig
    ) -> None:
        """Test export creates aider config."""
        exporter = AiderExporter()
        output_file = tmp_path / ".aider.conf.yml"

        result = exporter.export([sample_model], output_path=output_file)

        assert result == output_file
        config = yaml.safe_load(output_file.read_text())
        assert config["model"] == "openai/test-model-1"
        assert config["openai-api-key"] == "test-api-key-123"
        assert config["openai-api-base"] == "https://api.example.com/v1"

    def test_export_multiple_models_creates_aliases(
        self, tmp_path: Path, sample_models: list[ModelConfig]
    ) -> None:
        """Test export with multiple models creates aliases."""
        exporter = AiderExporter()
        output_file = tmp_path / ".aider.conf.yml"

        exporter.export(sample_models, output_path=output_file)

        config = yaml.safe_load(output_file.read_text())
        assert "alias" in config
        assert len(config["alias"]) == 1
        assert "Second Model:openai/test-model-2" in config["alias"]
        assert config["weak-model"] == "openai/test-model-2"

    def test_export_single_model_no_aliases(self, tmp_path: Path) -> None:
        """Test export with single model has no aliases."""
        exporter = AiderExporter()
        output_file = tmp_path / ".aider.conf.yml"
        model = ModelConfig(name="M", model="m", api_base="b", api_key="k")

        exporter.export([model], output_path=output_file)

        config = yaml.safe_load(output_file.read_text())
        assert "alias" not in config
        assert "weak-model" not in config

    def test_export_empty_models(self, tmp_path: Path) -> None:
        """Test export with empty models list."""
        exporter = AiderExporter()
        output_file = tmp_path / ".aider.conf.yml"

        exporter.export([], output_path=output_file)

        config = yaml.safe_load(output_file.read_text())
        assert config == {}

    def test_export_merge_existing(self, tmp_path: Path) -> None:
        """Test export merges with existing YAML."""
        exporter = AiderExporter()
        output_file = tmp_path / ".aider.conf.yml"

        existing = {"editor-model": "some-model", "dark-mode": True}
        output_file.write_text(yaml.dump(existing))

        model = ModelConfig(name="M", model="m", api_base="b", api_key="k")
        exporter.export([model], output_path=output_file, merge=True)

        config = yaml.safe_load(output_file.read_text())
        assert config["editor-model"] == "some-model"
        assert config["dark-mode"] is True
        assert config["model"] == "openai/m"


class TestCursorExporter:
    """Tests for Cursor exporter."""

    def test_tool_name(self) -> None:
        """Test tool_name property."""
        exporter = CursorExporter()
        assert exporter.tool_name == "cursor"

    def test_config_path(self) -> None:
        """Test config_path property."""
        exporter = CursorExporter()
        with patch.object(Path, "home", return_value=Path("/mock/home")):
            path = exporter.config_path
            assert path == Path("/mock/home/.cursor/.env")

    def test_export_creates_env_file(
        self, tmp_path: Path, sample_model: ModelConfig
    ) -> None:
        """Test export creates .env file."""
        exporter = CursorExporter()
        output_file = tmp_path / ".env"

        result = exporter.export([sample_model], output_path=output_file)

        assert result == output_file
        content = output_file.read_text()
        assert "OPENAI_API_KEY=test-api-key-123" in content
        assert "OPENAI_BASE_URL=https://api.example.com/v1" in content

    def test_export_empty_models(self, tmp_path: Path) -> None:
        """Test export with empty models list."""
        exporter = CursorExporter()
        output_file = tmp_path / ".env"

        exporter.export([], output_path=output_file)

        content = output_file.read_text()
        assert "OPENAI_API_KEY" not in content
        assert "OPENAI_BASE_URL" not in content

    def test_export_merge_existing(self, tmp_path: Path) -> None:
        """Test export merges with existing .env file."""
        exporter = CursorExporter()
        output_file = tmp_path / ".env"

        output_file.write_text("EXISTING_VAR=value\nOPENAI_API_KEY=old-key\n")
        model = ModelConfig(name="M", model="m", api_base="b", api_key="new-key")
        exporter.export([model], output_path=output_file, merge=True)

        content = output_file.read_text()
        assert "EXISTING_VAR=value" in content
        assert "OPENAI_API_KEY=new-key" in content
        assert "old-key" not in content

    def test_export_no_merge_overwrites(self, tmp_path: Path) -> None:
        """Test export without merge overwrites entire file."""
        exporter = CursorExporter()
        output_file = tmp_path / ".env"

        output_file.write_text("EXISTING_VAR=value\n")
        model = ModelConfig(name="M", model="m", api_base="b", api_key="k")
        exporter.export([model], output_path=output_file, merge=False)

        content = output_file.read_text()
        assert "EXISTING_VAR" not in content
        assert "OPENAI_API_KEY=k" in content


class TestGenericOpenAIExporter:
    """Tests for Generic OpenAI exporter."""

    def test_tool_name(self) -> None:
        """Test tool_name property."""
        exporter = GenericOpenAIExporter()
        assert exporter.tool_name == "generic-openai"

    def test_config_path(self) -> None:
        """Test config_path property."""
        exporter = GenericOpenAIExporter()
        with patch.object(Path, "home", return_value=Path("/mock/home")):
            path = exporter.config_path
            assert path == Path(
                "/mock/home/.config/hellmholtz/openai-compatible.json"
            )

    def test_export_creates_json(
        self, tmp_path: Path, sample_model: ModelConfig
    ) -> None:
        """Test export creates JSON config."""
        exporter = GenericOpenAIExporter()
        output_file = tmp_path / "config.json"

        result = exporter.export([sample_model], output_path=output_file)

        assert result == output_file
        config = json.loads(output_file.read_text())
        assert config["api_base"] == "https://api.example.com/v1"
        assert config["api_key"] == "test-api-key-123"
        assert len(config["models"]) == 1
        assert config["models"][0]["name"] == "Test Model"

    def test_export_multiple_models(
        self, tmp_path: Path, sample_models: list[ModelConfig]
    ) -> None:
        """Test export with multiple models."""
        exporter = GenericOpenAIExporter()
        output_file = tmp_path / "config.json"

        exporter.export(sample_models, output_path=output_file)

        config = json.loads(output_file.read_text())
        assert len(config["models"]) == 2

    def test_export_empty_models(self, tmp_path: Path) -> None:
        """Test export with empty models list."""
        exporter = GenericOpenAIExporter()
        output_file = tmp_path / "config.json"

        exporter.export([], output_path=output_file)

        config = json.loads(output_file.read_text())
        assert config == {}

    def test_export_merge_existing(self, tmp_path: Path) -> None:
        """Test export merges with existing config."""
        exporter = GenericOpenAIExporter()
        output_file = tmp_path / "config.json"

        existing = {"custom_field": "value", "models": []}
        output_file.write_text(json.dumps(existing))

        model = ModelConfig(name="M", model="m", api_base="b", api_key="k")
        exporter.export([model], output_path=output_file, merge=True)

        config = json.loads(output_file.read_text())
        assert config["custom_field"] == "value"
        assert config["api_key"] == "k"


class TestHermesAgentExporter:
    """Tests for Hermes Agent exporter."""

    def test_tool_name(self) -> None:
        """Test tool_name property."""
        exporter = HermesAgentExporter()
        assert exporter.tool_name == "hermes"

    def test_config_path(self) -> None:
        """Test config_path property."""
        exporter = HermesAgentExporter()
        with patch.object(Path, "home", return_value=Path("/mock/home")):
            path = exporter.config_path
            assert path == Path("/mock/home/.hermes/config.json")

    def test_export_creates_json(
        self, tmp_path: Path, sample_model: ModelConfig
    ) -> None:
        """Test export creates Hermes config."""
        exporter = HermesAgentExporter()
        output_file = tmp_path / "config.json"

        result = exporter.export([sample_model], output_path=output_file)

        assert result == output_file
        config = json.loads(output_file.read_text())
        assert config["provider"] == "custom"
        assert config["base_url"] == "https://api.example.com/v1"
        assert config["api_key"] == "test-api-key-123"
        assert config["model"] == "test-model-1"
        assert config["display_name"] == "Blablador"

    def test_export_models_list(
        self, tmp_path: Path, sample_models: list[ModelConfig]
    ) -> None:
        """Test export includes all models in list."""
        exporter = HermesAgentExporter()
        output_file = tmp_path / "config.json"

        exporter.export(sample_models, output_path=output_file)

        config = json.loads(output_file.read_text())
        assert len(config["models"]) == 2
        assert config["models"][0]["id"] == "test-model-1"
        assert config["models"][1]["id"] == "test-model-2"

    def test_export_empty_models(self, tmp_path: Path) -> None:
        """Test export with empty models list."""
        exporter = HermesAgentExporter()
        output_file = tmp_path / "config.json"

        exporter.export([], output_path=output_file)

        config = json.loads(output_file.read_text())
        assert config == {}

    def test_export_merge_existing(self, tmp_path: Path) -> None:
        """Test export merges with existing config."""
        exporter = HermesAgentExporter()
        output_file = tmp_path / "config.json"

        existing = {"custom_key": "value"}
        output_file.write_text(json.dumps(existing))

        model = ModelConfig(name="M", model="m", api_base="b", api_key="k")
        exporter.export([model], output_path=output_file, merge=True)

        config = json.loads(output_file.read_text())
        assert config["custom_key"] == "value"
        assert config["model"] == "m"


class TestJanAIExporter:
    """Tests for Jan.AI exporter."""

    def test_tool_name(self) -> None:
        """Test tool_name property."""
        exporter = JanAIExporter()
        assert exporter.tool_name == "jan"

    def test_config_path(self) -> None:
        """Test config_path property."""
        exporter = JanAIExporter()
        with patch.object(Path, "home", return_value=Path("/mock/home")):
            path = exporter.config_path
            assert path == Path("/mock/home/.config/jan/models/blablador.json")

    def test_export_creates_json(
        self, tmp_path: Path, sample_model: ModelConfig
    ) -> None:
        """Test export creates Jan.AI config."""
        exporter = JanAIExporter()
        output_file = tmp_path / "blablador.json"

        result = exporter.export([sample_model], output_path=output_file)

        assert result == output_file
        config = json.loads(output_file.read_text())
        assert config["id"] == "blablador"
        assert config["type"] == "openai"
        assert config["name"] == "Blablador"
        assert config["base_url"] == "https://api.example.com/v1"
        assert config["api_key"] == "test-api-key-123"
        assert len(config["models"]) == 1
        assert config["models"][0]["id"] == "test-model-1"
        assert config["models"][0]["context_length"] == 8192

    def test_export_model_without_context_length(self, tmp_path: Path) -> None:
        """Test export with no context length uses default."""
        exporter = JanAIExporter()
        output_file = tmp_path / "blablador.json"
        model = ModelConfig(name="M", model="m", api_base="b", api_key="k")

        exporter.export([model], output_path=output_file)

        config = json.loads(output_file.read_text())
        assert config["models"][0]["context_length"] == 98304

    def test_export_empty_models(self, tmp_path: Path) -> None:
        """Test export with empty models list."""
        exporter = JanAIExporter()
        output_file = tmp_path / "blablador.json"

        exporter.export([], output_path=output_file)

        config = json.loads(output_file.read_text())
        assert config == {}

    def test_export_merge_existing(self, tmp_path: Path) -> None:
        """Test export merges with existing config."""
        exporter = JanAIExporter()
        output_file = tmp_path / "blablador.json"

        existing = {"custom_key": "value"}
        output_file.write_text(json.dumps(existing))

        model = ModelConfig(name="M", model="m", api_base="b", api_key="k")
        exporter.export([model], output_path=output_file, merge=True)

        config = json.loads(output_file.read_text())
        assert config["custom_key"] == "value"
        assert config["id"] == "blablador"


class TestLangChainExporter:
    """Tests for LangChain exporter."""

    def test_tool_name(self) -> None:
        """Test tool_name property."""
        exporter = LangChainExporter()
        assert exporter.tool_name == "langchain"

    def test_config_path(self) -> None:
        """Test config_path property."""
        exporter = LangChainExporter()
        with patch.object(Path, "home", return_value=Path("/mock/home")):
            path = exporter.config_path
            assert path == Path("/mock/home/.config/hellmholtz/langchain.env")

    def test_export_creates_env_file(
        self, tmp_path: Path, sample_model: ModelConfig
    ) -> None:
        """Test export creates .env file."""
        exporter = LangChainExporter()
        output_file = tmp_path / "langchain.env"

        result = exporter.export([sample_model], output_path=output_file)

        assert result == output_file
        content = output_file.read_text()
        assert "OPENAI_API_KEY=test-api-key-123" in content
        assert "OPENAI_API_BASE=https://api.example.com/v1" in content
        assert "# Default model: test-model-1" in content

    def test_export_empty_models(self, tmp_path: Path) -> None:
        """Test export with empty models list."""
        exporter = LangChainExporter()
        output_file = tmp_path / "langchain.env"

        exporter.export([], output_path=output_file)

        content = output_file.read_text()
        assert "OPENAI_API_KEY" not in content
        assert "OPENAI_API_BASE" not in content

    def test_export_merge_existing(self, tmp_path: Path) -> None:
        """Test export merges with existing .env file."""
        exporter = LangChainExporter()
        output_file = tmp_path / "langchain.env"

        output_file.write_text("EXISTING_VAR=value\nOPENAI_API_KEY=old-key\n")
        model = ModelConfig(name="M", model="m", api_base="b", api_key="new-key")
        exporter.export([model], output_path=output_file, merge=True)

        content = output_file.read_text()
        assert "EXISTING_VAR=value" in content
        assert "OPENAI_API_KEY=new-key" in content
        assert "old-key" not in content

    def test_export_no_merge(self, tmp_path: Path) -> None:
        """Test export without merge overwrites file."""
        exporter = LangChainExporter()
        output_file = tmp_path / "langchain.env"

        output_file.write_text("EXISTING_VAR=value\n")
        model = ModelConfig(name="M", model="m", api_base="b", api_key="k")
        exporter.export([model], output_path=output_file, merge=False)

        content = output_file.read_text()
        assert "EXISTING_VAR" not in content
        assert "OPENAI_API_KEY=k" in content


class TestGPT4AllExporter:
    """Tests for GPT4All exporter."""

    def test_tool_name(self) -> None:
        """Test tool_name property."""
        exporter = GPT4AllExporter()
        assert exporter.tool_name == "gpt4all"

    def test_config_path(self) -> None:
        """Test config_path property."""
        exporter = GPT4AllExporter()
        with patch.object(Path, "home", return_value=Path("/mock/home")):
            path = exporter.config_path
            assert path == Path(
                "/mock/home/.config/hellmholtz/gpt4all-reference.json"
            )

    def test_export_creates_json(
        self, tmp_path: Path, sample_model: ModelConfig
    ) -> None:
        """Test export creates GPT4All reference config."""
        exporter = GPT4AllExporter()
        output_file = tmp_path / "gpt4all.json"

        result = exporter.export([sample_model], output_path=output_file)

        assert result == output_file
        config = json.loads(output_file.read_text())
        assert config["provider"] == "openai-compatible"
        assert config["base_url"] == "https://api.example.com/v1"
        assert config["api_key"] == "test-api-key-123"
        assert config["default_model"] == "test-model-1"
        assert "setup_instructions" in config
        assert config["setup_instructions"]["step_3"] == "API Key: test-api-key-123"

    def test_export_available_models(
        self, tmp_path: Path, sample_models: list[ModelConfig]
    ) -> None:
        """Test export includes all available models."""
        exporter = GPT4AllExporter()
        output_file = tmp_path / "gpt4all.json"

        exporter.export(sample_models, output_path=output_file)

        config = json.loads(output_file.read_text())
        assert len(config["available_models"]) == 2

    def test_export_empty_models(self, tmp_path: Path) -> None:
        """Test export with empty models list."""
        exporter = GPT4AllExporter()
        output_file = tmp_path / "gpt4all.json"

        exporter.export([], output_path=output_file)

        config = json.loads(output_file.read_text())
        assert config == {}

    def test_export_merge_existing(self, tmp_path: Path) -> None:
        """Test export merges with existing config."""
        exporter = GPT4AllExporter()
        output_file = tmp_path / "gpt4all.json"

        existing = {"custom_key": "value"}
        output_file.write_text(json.dumps(existing))

        model = ModelConfig(name="M", model="m", api_base="b", api_key="k")
        exporter.export([model], output_path=output_file, merge=True)

        config = json.loads(output_file.read_text())
        assert config["custom_key"] == "value"
        assert config["default_model"] == "m"


class TestPiAgentExporter:
    """Tests for Pi Agent exporter."""

    def test_tool_name(self) -> None:
        """Test tool_name property."""
        exporter = PiAgentExporter()
        assert exporter.tool_name == "pi"

    def test_config_path(self) -> None:
        """Test config_path property."""
        exporter = PiAgentExporter()
        with patch.object(Path, "home", return_value=Path("/mock/home")):
            path = exporter.config_path
            assert path == Path("/mock/home/.pi/agent/models.json")

    def test_export_creates_json(
        self, tmp_path: Path, sample_model: ModelConfig
    ) -> None:
        """Test export creates Pi Agent config."""
        exporter = PiAgentExporter()
        output_file = tmp_path / "models.json"

        result = exporter.export([sample_model], output_path=output_file)

        assert result == output_file
        config = json.loads(output_file.read_text())
        assert "providers" in config
        assert "blablador" in config["providers"]
        assert config["providers"]["blablador"]["baseUrl"] == "https://api.example.com/v1"
        assert config["providers"]["blablador"]["apiKey"] == "test-api-key-123"
        assert config["providers"]["blablador"]["api"] == "openai-completions"
        assert len(config["providers"]["blablador"]["models"]) == 1
        assert config["providers"]["blablador"]["models"][0]["id"] == "test-model-1"

    def test_export_multiple_models(
        self, tmp_path: Path, sample_models: list[ModelConfig]
    ) -> None:
        """Test export with multiple models."""
        exporter = PiAgentExporter()
        output_file = tmp_path / "models.json"

        exporter.export(sample_models, output_path=output_file)

        config = json.loads(output_file.read_text())
        assert len(config["providers"]["blablador"]["models"]) == 2

    def test_export_empty_models(self, tmp_path: Path) -> None:
        """Test export with empty models list."""
        exporter = PiAgentExporter()
        output_file = tmp_path / "models.json"

        exporter.export([], output_path=output_file)

        config = json.loads(output_file.read_text())
        assert config == {}

    def test_export_merge_existing(self, tmp_path: Path) -> None:
        """Test export merges with existing config."""
        exporter = PiAgentExporter()
        output_file = tmp_path / "models.json"

        existing = {"providers": {"other_provider": {"models": []}}}
        output_file.write_text(json.dumps(existing))

        model = ModelConfig(name="M", model="m", api_base="b", api_key="k")
        exporter.export([model], output_path=output_file, merge=True)

        config = json.loads(output_file.read_text())
        assert "other_provider" in config["providers"]
        assert "blablador" in config["providers"]


class TestExporterRegistry:
    """Tests for the EXPORTERS registry and helper functions."""

    def test_get_exporter_valid(self) -> None:
        """Test get_exporter with valid tool names."""
        for tool_name in EXPORTERS.keys():
            exporter = get_exporter(tool_name)
            assert exporter.tool_name == tool_name

    def test_get_exporter_invalid(self) -> None:
        """Test get_exporter with invalid tool name."""
        with pytest.raises(ValueError, match="Unsupported tool"):
            get_exporter("nonexistent-tool")

    def test_list_exporters(self) -> None:
        """Test list_exporters returns all 11 exporters."""
        exporters = list_exporters()
        assert len(exporters) == 11
        expected = [
            "opencode",
            "claude-code",
            "continue",
            "aider",
            "cursor",
            "generic-openai",
            "hermes",
            "jan",
            "langchain",
            "gpt4all",
            "pi",
        ]
        for name in expected:
            assert name in exporters

    def test_all_exporters_in_registry(self) -> None:
        """Test that all concrete exporter classes are in registry."""
        all_exporters = [
            OpenCodeExporter,
            ClaudeCodeExporter,
            ContinueExporter,
            AiderExporter,
            CursorExporter,
            GenericOpenAIExporter,
            HermesAgentExporter,
            JanAIExporter,
            LangChainExporter,
            GPT4AllExporter,
            PiAgentExporter,
        ]
        for exporter_cls in all_exporters:
            found = False
            for cls in EXPORTERS.values():
                if cls is exporter_cls:
                    found = True
                    break
            assert found, f"{exporter_cls.__name__} not in EXPORTERS registry"


class TestExportWithDefaults:
    """Tests for exporters using default config_path when no output_path given."""

    def test_opencode_export_default_path(self, tmp_path: Path) -> None:
        """Test OpenCode export uses config_path when output_path is None."""
        exporter = OpenCodeExporter()
        model = ModelConfig(name="M", model="m", api_base="b", api_key="k")

        with patch.object(Path, "home", return_value=tmp_path):
            result = exporter.export([model])

        assert result == tmp_path / ".config" / "opencode" / "opencode.json"
        assert result.exists()

    def test_claude_code_export_default_path(self, tmp_path: Path) -> None:
        """Test Claude Code export uses config_path when output_path is None."""
        exporter = ClaudeCodeExporter()
        model = ModelConfig(name="M", model="m", api_base="b", api_key="k")

        with patch.object(Path, "home", return_value=tmp_path):
            result = exporter.export([model])

        assert result == tmp_path / ".claude" / "settings.json"
        assert result.exists()

    def test_continue_export_default_path(self, tmp_path: Path) -> None:
        """Test Continue export uses config_path when output_path is None."""
        exporter = ContinueExporter()
        model = ModelConfig(name="M", model="m", api_base="b", api_key="k")

        with patch.object(Path, "home", return_value=tmp_path):
            result = exporter.export([model])

        assert result == tmp_path / ".continue" / "config.yaml"
        assert result.exists()

    def test_aider_export_default_path(self, tmp_path: Path) -> None:
        """Test Aider export uses config_path when output_path is None."""
        exporter = AiderExporter()
        model = ModelConfig(name="M", model="m", api_base="b", api_key="k")

        with patch.object(Path, "home", return_value=tmp_path):
            result = exporter.export([model])

        assert result == tmp_path / ".aider.conf.yml"
        assert result.exists()

    def test_cursor_export_default_path(self, tmp_path: Path) -> None:
        """Test Cursor export uses config_path when output_path is None."""
        exporter = CursorExporter()
        model = ModelConfig(name="M", model="m", api_base="b", api_key="k")

        with patch.object(Path, "home", return_value=tmp_path):
            result = exporter.export([model])

        assert result == tmp_path / ".cursor" / ".env"
        assert result.exists()

    def test_generic_openai_export_default_path(self, tmp_path: Path) -> None:
        """Test Generic OpenAI export uses config_path when output_path is None."""
        exporter = GenericOpenAIExporter()
        model = ModelConfig(name="M", model="m", api_base="b", api_key="k")

        with patch.object(Path, "home", return_value=tmp_path):
            result = exporter.export([model])

        assert result == tmp_path / ".config" / "hellmholtz" / "openai-compatible.json"
        assert result.exists()

    def test_hermes_export_default_path(self, tmp_path: Path) -> None:
        """Test Hermes export uses config_path when output_path is None."""
        exporter = HermesAgentExporter()
        model = ModelConfig(name="M", model="m", api_base="b", api_key="k")

        with patch.object(Path, "home", return_value=tmp_path):
            result = exporter.export([model])

        assert result == tmp_path / ".hermes" / "config.json"
        assert result.exists()

    def test_jan_export_default_path(self, tmp_path: Path) -> None:
        """Test Jan.AI export uses config_path when output_path is None."""
        exporter = JanAIExporter()
        model = ModelConfig(name="M", model="m", api_base="b", api_key="k")

        with patch.object(Path, "home", return_value=tmp_path):
            result = exporter.export([model])

        assert result == tmp_path / ".config" / "jan" / "models" / "blablador.json"
        assert result.exists()

    def test_langchain_export_default_path(self, tmp_path: Path) -> None:
        """Test LangChain export uses config_path when output_path is None."""
        exporter = LangChainExporter()
        model = ModelConfig(name="M", model="m", api_base="b", api_key="k")

        with patch.object(Path, "home", return_value=tmp_path):
            result = exporter.export([model])

        assert result == tmp_path / ".config" / "hellmholtz" / "langchain.env"
        assert result.exists()

    def test_gpt4all_export_default_path(self, tmp_path: Path) -> None:
        """Test GPT4All export uses config_path when output_path is None."""
        exporter = GPT4AllExporter()
        model = ModelConfig(name="M", model="m", api_base="b", api_key="k")

        with patch.object(Path, "home", return_value=tmp_path):
            result = exporter.export([model])

        assert result == tmp_path / ".config" / "hellmholtz" / "gpt4all-reference.json"
        assert result.exists()

    def test_pi_export_default_path(self, tmp_path: Path) -> None:
        """Test Pi Agent export uses config_path when output_path is None."""
        exporter = PiAgentExporter()
        model = ModelConfig(name="M", model="m", api_base="b", api_key="k")

        with patch.object(Path, "home", return_value=tmp_path):
            result = exporter.export([model])

        assert result == tmp_path / ".pi" / "agent" / "models.json"
        assert result.exists()


class TestExportMergeBehavior:
    """Tests specifically for merge behavior across exporters."""

    def test_merge_preserves_unrelated_keys_json(self, tmp_path: Path) -> None:
        """Test JSON exporter merge preserves unrelated config keys."""
        exporter = GenericOpenAIExporter()
        output_file = tmp_path / "config.json"

        existing = {"existing_key": "value", "nested": {"inner": True}}
        output_file.write_text(json.dumps(existing))

        model = ModelConfig(name="M", model="m", api_base="b", api_key="k")
        exporter.export([model], output_path=output_file, merge=True)

        config = json.loads(output_file.read_text())
        assert config["existing_key"] == "value"
        assert config["nested"]["inner"] is True
        assert config["api_key"] == "k"

    def test_merge_preserves_unrelated_keys_yaml(self, tmp_path: Path) -> None:
        """Test YAML exporter merge preserves unrelated config keys."""
        exporter = ContinueExporter()
        output_file = tmp_path / "config.yaml"

        existing = {"existing_key": "value", "custom_list": [1, 2, 3]}
        output_file.write_text(yaml.dump(existing))

        model = ModelConfig(name="M", model="m", api_base="b", api_key="k")
        exporter.export([model], output_path=output_file, merge=True)

        config = yaml.safe_load(output_file.read_text())
        assert config["existing_key"] == "value"
        assert config["custom_list"] == [1, 2, 3]
        assert len(config["models"]) == 1

    def test_merge_false_ignores_existing(self, tmp_path: Path) -> None:
        """Test merge=False ignores existing config entirely."""
        exporter = GenericOpenAIExporter()
        output_file = tmp_path / "config.json"

        existing = {"existing_key": "value", "api_key": "old"}
        output_file.write_text(json.dumps(existing))

        model = ModelConfig(name="M", model="m", api_base="b", api_key="k")
        exporter.export([model], output_path=output_file, merge=False)

        config = json.loads(output_file.read_text())
        assert "existing_key" not in config
        assert config["api_key"] == "k"


class TestExportCreatesParentDirectories:
    """Tests that exporters create parent directories when needed."""

    def test_opencode_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Test OpenCode exporter creates parent directories."""
        exporter = OpenCodeExporter()
        output_file = tmp_path / "deep" / "nested" / "opencode.json"
        model = ModelConfig(name="M", model="m", api_base="b", api_key="k")

        exporter.export([model], output_path=output_file)

        assert output_file.exists()
        assert output_file.parent.exists()

    def test_cursor_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Test Cursor exporter creates parent directories."""
        exporter = CursorExporter()
        output_file = tmp_path / "deep" / "nested" / ".env"
        model = ModelConfig(name="M", model="m", api_base="b", api_key="k")

        exporter.export([model], output_path=output_file)

        assert output_file.exists()
        assert output_file.parent.exists()


class TestExportModelConfigToDict:
    """Tests for ModelConfig.to_dict used in GenericOpenAIExporter."""

    def test_model_config_to_dict_with_all_fields(
        self, sample_model: ModelConfig
    ) -> None:
        """Test to_dict includes all provided fields."""
        result = sample_model.to_dict()
        assert result["name"] == "Test Model"
        assert result["provider"] == "blablador"
        assert result["model"] == "test-model-1"
        assert result["apiBase"] == "https://api.example.com/v1"
        assert result["apiKey"] == "test-api-key-123"
        assert result["contextLength"] == 8192
        assert result["maxTokens"] == 4096
        assert result["roles"] == ["chat"]

    def test_model_config_to_dict_minimal(self) -> None:
        """Test to_dict with minimal fields."""
        model = ModelConfig(name="M")
        result = model.to_dict()
        assert result["name"] == "M"
        assert result["provider"] == "blablador"
        assert result["model"] == ""
        assert "apiBase" not in result
        assert "apiKey" not in result
        assert "contextLength" not in result
        assert "maxTokens" not in result
