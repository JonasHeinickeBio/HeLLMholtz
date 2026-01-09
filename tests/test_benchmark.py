import json
from pathlib import Path
from unittest.mock import MagicMock, patch
from hellmholtz.benchmark import run_benchmarks
from hellmholtz.core.prompts import Prompt, Message

@patch("hellmholtz.benchmark.runner.chat_raw")
def test_benchmark_run(mock_chat_raw: object, tmp_path) -> None:
    # Setup mock response
    mock_response = MagicMock()
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 20
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Test response"
    mock_chat_raw.return_value = mock_response

    models = ["openai:gpt-4o"]
    prompts = [
        Prompt(
            id="test-1",
            category="test",
            messages=[Message(role="user", content="test prompt")]
        )
    ]

    results = run_benchmarks(
        models=models,
        prompts=prompts,
        results_dir=str(tmp_path),
        temperatures=[0.1],  # Only test one temperature to keep it simple
        replications=1
    )

    assert len(results) == 1
    assert results[0].success is True
    assert results[0].input_tokens == 10

    # Check file created
    files = list(tmp_path.glob("benchmark_*.json"))
    assert len(files) == 1

    with open(files[0]) as f:
        data = json.load(f)
        assert len(data) == 1
        assert data[0]["model"] == "openai:gpt-4o"
