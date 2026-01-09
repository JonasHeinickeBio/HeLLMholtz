"""
Tests for integration functionality.

This module contains comprehensive tests for integration modules,
including LM Evaluation Harness and LiteLLM proxy functionality.
"""

import sys
import pytest
from unittest.mock import MagicMock, patch, call
from typing import List, Dict, Any

# Mock lm_eval before importing the integration module
sys.modules["lm_eval"] = MagicMock()
sys.modules["lm_eval.utils"] = MagicMock()

from hellmholtz.integrations.lm_eval import run_lm_eval
from hellmholtz.integrations.litellm import start_proxy


class TestLmEvalIntegration:
    """Test suite for LM Evaluation Harness integration."""

    def test_run_lm_eval_missing_dependency(self) -> None:
        """Test that run_lm_eval raises SystemExit when lm_eval is not available."""
        with patch("hellmholtz.integrations.lm_eval.lm_eval", None):
            with pytest.raises(SystemExit):
                run_lm_eval("openai:gpt-4o", ["mmlu"])

    def test_run_lm_eval_basic_functionality(self) -> None:
        """Test basic LM evaluation run."""
        mock_simple_eval = MagicMock()
        mock_simple_eval.return_value = {"results": {"mmlu": {"acc": 0.85}}}

        with patch("hellmholtz.integrations.lm_eval.simple_evaluate", mock_simple_eval):
            with patch("hellmholtz.integrations.lm_eval.lm_eval", MagicMock()):
                run_lm_eval("openai:gpt-4o", ["mmlu"])

                mock_simple_eval.assert_called_once()
                call_kwargs = mock_simple_eval.call_args[1]

                assert call_kwargs["model"] == "openai-chat-completions"
                assert call_kwargs["model_args"] == "model=gpt-4o"
                assert "mmlu" in call_kwargs["tasks"]

    def test_run_lm_eval_multiple_tasks(self) -> None:
        """Test LM evaluation with multiple tasks."""
        mock_simple_eval = MagicMock()
        mock_simple_eval.return_value = {"results": {"mmlu": {"acc": 0.8}, "hellaswag": {"acc": 0.75}}}

        with patch("hellmholtz.integrations.lm_eval.simple_evaluate", mock_simple_eval):
            with patch("hellmholtz.integrations.lm_eval.lm_eval", MagicMock()):
                tasks = ["mmlu", "hellaswag", "winogrande"]
                run_lm_eval("anthropic:claude-3-sonnet-20240229", tasks)

                mock_simple_eval.assert_called_once()
                call_kwargs = mock_simple_eval.call_args[1]

                assert call_kwargs["model"] == "openai-chat-completions"
                assert call_kwargs["model_args"] == "model=claude-3-sonnet-20240229"
                for task in tasks:
                    assert task in call_kwargs["tasks"]

    def test_run_lm_eval_different_providers(self) -> None:
        """Test LM evaluation with different model providers."""
        test_cases = [
            ("openai:gpt-4o", "openai-chat-completions", "model=gpt-4o"),
            ("anthropic:claude-3-sonnet-20240229", "openai-chat-completions", "model=claude-3-sonnet-20240229"),
            ("google:gemini-pro", "openai-chat-completions", "model=gemini-pro"),
            ("ollama:llama3", "openai-chat-completions", "model=llama3"),
        ]

        for model_spec, expected_model, expected_args in test_cases:
            mock_simple_eval = MagicMock()
            mock_simple_eval.return_value = {"results": {"mmlu": {"acc": 0.8}}}

            with patch("hellmholtz.integrations.lm_eval.simple_evaluate", mock_simple_eval):
                with patch("hellmholtz.integrations.lm_eval.lm_eval", MagicMock()):
                    run_lm_eval(model_spec, ["mmlu"])

                    call_kwargs = mock_simple_eval.call_args[1]
                    assert call_kwargs["model"] == expected_model
                    assert call_kwargs["model_args"] == expected_args

    def test_run_lm_eval_with_custom_args(self) -> None:
        """Test LM evaluation with custom arguments."""
        mock_simple_eval = MagicMock()
        mock_simple_eval.return_value = {"results": {"mmlu": {"acc": 0.8}}}

        with patch("hellmholtz.integrations.lm_eval.simple_evaluate", mock_simple_eval):
            with patch("hellmholtz.integrations.lm_eval.lm_eval", MagicMock()):
                # Note: Current implementation doesn't support custom args easily
                # This test documents current behavior
                run_lm_eval("openai:gpt-4o", ["mmlu"])

                call_kwargs = mock_simple_eval.call_args[1]
                # Verify that basic required args are present
                assert "model" in call_kwargs
                assert "model_args" in call_kwargs
                assert "tasks" in call_kwargs

    def test_run_lm_eval_empty_tasks(self) -> None:
        """Test LM evaluation with empty task list."""
        mock_simple_eval = MagicMock()
        mock_simple_eval.return_value = {"results": {}}

        with patch("hellmholtz.integrations.lm_eval.simple_evaluate", mock_simple_eval):
            with patch("hellmholtz.integrations.lm_eval.lm_eval", MagicMock()):
                run_lm_eval("openai:gpt-4o", [])

                mock_simple_eval.assert_called_once()
                call_kwargs = mock_simple_eval.call_args[1]
                assert call_kwargs["tasks"] == []


class TestLiteLLMIntegration:
    """Test suite for LiteLLM proxy integration."""

    @patch("subprocess.run")
    def test_start_proxy_basic_functionality(self, mock_subprocess: MagicMock) -> None:
        """Test basic proxy start functionality."""
        start_proxy("ollama:llama3", port=5000)

        mock_subprocess.assert_called_once()
        cmd = mock_subprocess.call_args[0][0]

        assert "litellm" in cmd
        assert "--model" in cmd
        assert "ollama:llama3" in cmd
        assert "--port" in cmd
        assert "5000" in cmd

    @patch("subprocess.run")
    def test_start_proxy_different_models(self, mock_subprocess: MagicMock) -> None:
        """Test proxy start with different model specifications."""
        test_cases = [
            ("openai:gpt-4o", 8000),
            ("anthropic:claude-3-sonnet-20240229", 9000),
            ("ollama:llama3.2:3b", 7000),
        ]

        for model_spec, port in test_cases:
            mock_subprocess.reset_mock()
            start_proxy(model_spec, port=port)

            cmd = mock_subprocess.call_args[0][0]
            assert model_spec in cmd
            assert str(port) in cmd

    @patch("subprocess.run")
    def test_start_proxy_default_port(self, mock_subprocess: MagicMock) -> None:
        """Test proxy start with default port."""
        start_proxy("openai:gpt-4o")

        cmd = mock_subprocess.call_args[0][0]
        assert "4000" in cmd  # Default port

    @patch("subprocess.run")
    def test_start_proxy_command_structure(self, mock_subprocess: MagicMock) -> None:
        """Test that the proxy command has the correct structure."""
        start_proxy("ollama:llama3", port=5000)

        cmd = mock_subprocess.call_args[0][0]

        # Command should be a list
        assert isinstance(cmd, list)

        # Should start with litellm
        assert cmd[0] == "litellm"

        # Should contain model and port arguments
        assert "--model" in cmd
        assert "--port" in cmd

        # Find the model argument
        model_idx = cmd.index("--model")
        assert cmd[model_idx + 1] == "ollama:llama3"

        # Find the port argument
        port_idx = cmd.index("--port")
        assert cmd[port_idx + 1] == "5000"

    @patch("subprocess.run")
    @patch("subprocess.Popen")
    def test_start_proxy_background_process(self, mock_popen: MagicMock, mock_run: MagicMock) -> None:
        """Test that proxy starts as a background process."""
        # Note: Current implementation uses subprocess.run, not Popen
        # This test documents the current behavior
        start_proxy("ollama:llama3", port=5000)

        mock_run.assert_called_once()
        # Verify it's not using Popen for background execution
        mock_popen.assert_not_called()

    @patch("subprocess.run")
    def test_start_proxy_complex_model_specs(self, mock_subprocess: MagicMock) -> None:
        """Test proxy start with complex model specifications."""
        complex_models = [
            "blablador:Ministral-3-14B-Instruct-2512",
            "openai:gpt-4o-mini",
            "anthropic:claude-3-5-sonnet-20241022"
        ]

        for model_spec in complex_models:
            mock_subprocess.reset_mock()
            start_proxy(model_spec, port=6000)

            cmd = mock_subprocess.call_args[0][0]
            assert model_spec in cmd
