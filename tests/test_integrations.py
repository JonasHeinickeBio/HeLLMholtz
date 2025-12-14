import sys
from unittest.mock import MagicMock, patch
import pytest

# Mock lm_eval before importing the integration module
sys.modules["lm_eval"] = MagicMock()
sys.modules["lm_eval.utils"] = MagicMock()
from hellmholtz.integrations.lm_eval import run_lm_eval
from hellmholtz.integrations.litellm import start_proxy

def test_run_lm_eval_missing_dep():
    # Simulate missing dependency by setting lm_eval to None in the module
    with patch("hellmholtz.integrations.lm_eval.lm_eval", None):
        with pytest.raises(SystemExit):
            run_lm_eval("openai:gpt-4o", ["mmlu"])

def test_run_lm_eval():
    # Setup the mock for simple_evaluate
    mock_simple_eval = MagicMock()
    mock_simple_eval.return_value = {"results": "test"}

    # We need to patch where it's imported in our module
    # Since we mocked sys.modules["lm_eval"], the import in the module succeeded
    # and simple_evaluate is bound to the mock's attribute.
    # However, to control the return value easily, let's patch it in our module.

    with patch("hellmholtz.integrations.lm_eval.simple_evaluate", mock_simple_eval):
        # Ensure the module-level lm_eval is not None (it should be our mock from sys.modules)
        # But just in case, let's patch it to be sure.
        with patch("hellmholtz.integrations.lm_eval.lm_eval", MagicMock()):
            run_lm_eval("openai:gpt-4o", ["mmlu"])
            mock_simple_eval.assert_called_once()
            assert mock_simple_eval.call_args[1]["model"] == "openai-chat-completions"
            assert mock_simple_eval.call_args[1]["model_args"] == "model=gpt-4o"

@patch("subprocess.run")
def test_litellm_not_installed(mock_subprocess: object) -> None:
    start_proxy("ollama:llama3", port=5000)
    mock_subprocess.assert_called_once()
    cmd = mock_subprocess.call_args[0][0]
    assert "litellm" in cmd
    assert "--model" in cmd

@patch("subprocess.run")
def test_litellm_start(mock_subprocess: object) -> None:
    start_proxy("ollama:llama3", port=5000)
    mock_subprocess.assert_called_once()
    cmd = mock_subprocess.call_args[0][0]
    assert "litellm" in cmd
    assert "--model" in cmd
    assert "ollama:llama3" in cmd
    assert "--port" in cmd
    assert "5000" in cmd
