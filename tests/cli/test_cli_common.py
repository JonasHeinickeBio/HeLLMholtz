"""Tests for hellmholtz.cli.common module."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer

from hellmholtz.cli.common import (
    configure_logging,
    format_token_limit,
    generate_output_path,
    get_prompts_by_category_or_default,
    handle_error,
    load_prompts_from_file,
    parse_models,
    parse_temperatures,
    save_report_to_file,
)
from hellmholtz.core.prompts import Message, Prompt


class TestConfigureLogging:
    """Tests for configure_logging."""

    @patch("hellmholtz.cli.common.logging.basicConfig")
    def test_calls_basic_config(self, mock_basic: MagicMock) -> None:
        configure_logging()
        mock_basic.assert_called_once()
        kwargs = mock_basic.call_args[1]
        assert kwargs["level"] == 20  # logging.INFO

    @patch("hellmholtz.cli.common.logging.basicConfig")
    def test_format_contains_required_parts(self, mock_basic: MagicMock) -> None:
        configure_logging()
        fmt = mock_basic.call_args[1]["format"]
        assert "%(asctime)s" in fmt
        assert "%(name)s" in fmt
        assert "%(levelname)s" in fmt
        assert "%(message)s" in fmt


class TestFormatTokenLimit:
    """Tests for format_token_limit."""

    def test_small_value(self) -> None:
        assert format_token_limit(512) == "512"

    def test_zero(self) -> None:
        assert format_token_limit(0) == "0"

    def test_just_below_1024(self) -> None:
        assert format_token_limit(1023) == "1023"

    def test_exactly_1024(self) -> None:
        assert format_token_limit(1024) == "1k"

    def test_large_k_value(self) -> None:
        assert format_token_limit(8192) == "8k"

    def test_exactly_1048576(self) -> None:
        assert format_token_limit(1048576) == "1M"

    def test_multi_megabyte(self) -> None:
        assert format_token_limit(2097152) == "2M"

    def test_odd_k_value(self) -> None:
        assert format_token_limit(3072) == "3k"

    def test_none_value(self) -> None:
        assert format_token_limit(None) == "0"

    def test_huge_value(self) -> None:
        # 100 * 1024 * 1024 = 104857600 bytes = 100M
        assert format_token_limit(104857600) == "100M"


class TestHandleError:
    """Tests for handle_error."""

    def test_raises_exit_with_code_1(self) -> None:
        with pytest.raises(typer.Exit) as exc_info:
            handle_error(ValueError("bad"), "Something failed")
        assert exc_info.value.exit_code == 1

    def test_raises_exit_with_custom_code(self) -> None:
        with pytest.raises(typer.Exit) as exc_info:
            handle_error(RuntimeError("crash"), "Process failed", exit_code=2)
        assert exc_info.value.exit_code == 2

    @patch("hellmholtz.cli.common.logger")
    def test_logs_the_error(self, mock_logger: MagicMock) -> None:
        with pytest.raises(typer.Exit):
            handle_error(ValueError("oops"), "Test context")
        mock_logger.error.assert_called_once()
        assert "Test context" in mock_logger.error.call_args[0][0]
        assert "oops" in mock_logger.error.call_args[0][0]


class TestLoadPromptsFromFile:
    """Tests for load_prompts_from_file."""

    def test_load_json_single_prompt(self, tmp_path: Path) -> None:
        prompt_data = {
            "id": "p1",
            "category": "test",
            "messages": [{"role": "user", "content": "hello"}],
        }
        f = tmp_path / "prompt.json"
        f.write_text(json.dumps(prompt_data))
        prompts = load_prompts_from_file(f)
        assert len(prompts) == 1
        assert prompts[0].id == "p1"

    def test_load_json_list_of_prompts(self, tmp_path: Path) -> None:
        prompt_data = [
            {"id": "p1", "category": "test", "messages": [{"role": "user", "content": "a"}]},
            {"id": "p2", "category": "test", "messages": [{"role": "user", "content": "b"}]},
        ]
        f = tmp_path / "prompts.json"
        f.write_text(json.dumps(prompt_data))
        prompts = load_prompts_from_file(f)
        assert len(prompts) == 2

    def test_load_txt_prompts(self, tmp_path: Path) -> None:
        f = tmp_path / "prompts.txt"
        f.write_text("What is 2+2?\nExplain gravity.\n\nDescribe the sun.\n")
        prompts = load_prompts_from_file(f)
        assert len(prompts) == 3
        assert prompts[0].messages[0].content == "What is 2+2?"
        assert prompts[0].category == "custom"
        assert prompts[0].id.startswith("custom_")

    def test_load_txt_empty_file(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.txt"
        f.write_text("")
        prompts = load_prompts_from_file(f)
        assert len(prompts) == 0

    def test_load_json_invalid_format(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.json"
        f.write_text("not json {{{")
        with pytest.raises(typer.Exit):
            load_prompts_from_file(f)


class TestParseTemperatures:
    """Tests for parse_temperatures."""

    def test_valid_comma_separated(self) -> None:
        result = parse_temperatures("0.1,0.5,1.0")
        assert result == [0.1, 0.5, 1.0]

    def test_single_value(self) -> None:
        result = parse_temperatures("0.7")
        assert result == [0.7]

    def test_with_spaces(self) -> None:
        result = parse_temperatures(" 0.1 , 0.7 , 1.0 ")
        assert result == [0.1, 0.7, 1.0]

    def test_none_returns_defaults(self) -> None:
        result = parse_temperatures(None)
        assert result == [0.1, 0.7, 1.0]

    def test_invalid_value_raises_exit(self) -> None:
        with pytest.raises(typer.Exit):
            parse_temperatures("abc,def")


class TestGenerateOutputPath:
    """Tests for generate_output_path."""

    def test_markdown_format(self) -> None:
        path = generate_output_path([], "markdown", timestamp="2024-01-01T00-00-00")
        assert path.suffix == ".md"
        assert "2024-01-01T00-00-00" in str(path)

    def test_html_format(self) -> None:
        path = generate_output_path([], "html", timestamp="2024-01-01T00-00-00")
        assert path.suffix == ".html"
        assert "html" in str(path)

    def test_custom_base_dir(self) -> None:
        path = generate_output_path([], "markdown", timestamp="t", base_dir="out")
        assert str(path).startswith("out/")

    def test_no_timestamp_uses_result_timestamp(self) -> None:
        mock_result = MagicMock()
        mock_result.timestamp = "2024:06:01.123"
        path = generate_output_path([mock_result], "markdown")
        assert "2024-06-01-123" in str(path)

    def test_no_results_no_timestamp(self) -> None:
        path = generate_output_path([], "markdown")
        assert "unknown" in str(path)


class TestSaveReportToFile:
    """Tests for save_report_to_file."""

    def test_creates_file_with_content(self, tmp_path: Path) -> None:
        output = tmp_path / "sub" / "report.md"
        save_report_to_file("Hello World", output)
        assert output.exists()
        assert output.read_text() == "Hello World"

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        output = tmp_path / "a" / "b" / "c" / "report.md"
        save_report_to_file("nested", output)
        assert output.exists()

    def test_overwrites_existing_file(self, tmp_path: Path) -> None:
        output = tmp_path / "report.md"
        output.write_text("old")
        save_report_to_file("new", output)
        assert output.read_text() == "new"


class TestGetPromptsByCategoryOrDefault:
    """Tests for get_prompts_by_category_or_default."""

    def test_valid_category(self) -> None:
        prompts = get_prompts_by_category_or_default("reasoning")
        assert len(prompts) > 0

    def test_none_category_raises_exit(self) -> None:
        with pytest.raises(typer.Exit):
            get_prompts_by_category_or_default(None)

    def test_invalid_category_raises_exit(self) -> None:
        with patch(
            "hellmholtz.benchmark.prompts.get_prompts_by_category", return_value=[]
        ):
            with pytest.raises(typer.Exit):
                get_prompts_by_category_or_default("nonexistent_category_xyz")


class TestParseModels:
    """Tests for parse_models."""

    def test_with_explicit_models(self) -> None:
        result = parse_models("openai:gpt-4o,anthropic:claude-3")
        assert result == ["openai:gpt-4o", "anthropic:claude-3"]

    def test_blablador_fallback(self) -> None:
        with patch("hellmholtz.core.config.get_settings") as mock_settings:
            mock_settings.return_value.default_models = []
            with patch("hellmholtz.providers.blablador.list_models") as mock_list:
                mock_model = MagicMock()
                mock_model.name = "test-model"
                mock_model.id = "test-id"
                mock_list.return_value = [mock_model]
                result = parse_models(None)
                assert any("blablador:test-model" in m for m in result)

    def test_blablador_exception_handled(self) -> None:
        with patch("hellmholtz.core.config.get_settings") as mock_settings:
            mock_settings.return_value.default_models = ["openai:gpt-4o"]
            with patch(
                "hellmholtz.providers.blablador.list_models",
                side_effect=Exception("API error"),
            ):
                result = parse_models(None)
                assert "openai:gpt-4o" in result

    def test_no_models_raises_exit(self) -> None:
        with patch("hellmholtz.core.config.get_settings") as mock_settings:
            mock_settings.return_value.default_models = []
            with patch("hellmholtz.providers.blablador.list_models", return_value=[]):
                with pytest.raises(typer.Exit):
                    parse_models(None)
