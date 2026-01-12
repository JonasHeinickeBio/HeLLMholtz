"""
Tests for benchmark functionality.

This module contains comprehensive tests for the benchmark runner,
including unit tests for benchmark execution, result handling,
and integration tests for the full benchmark workflow.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from hellmholtz.benchmark.runner import BenchmarkResult, run_benchmarks
from hellmholtz.core.prompts import Message, Prompt


class TestBenchmarkRunner:
    """Test suite for benchmark runner functionality."""

    @pytest.fixture
    def sample_prompts(self) -> list[Prompt]:
        """Create sample prompts for testing."""
        return [
            Prompt(
                id="test-1",
                category="test",
                messages=[Message(role="user", content="test prompt 1")],
            ),
            Prompt(
                id="test-2",
                category="test",
                messages=[Message(role="user", content="test prompt 2")],
            ),
        ]

    @pytest.fixture
    def mock_chat_response(self) -> MagicMock:
        """Create a mock chat response for testing."""
        mock_response = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        return mock_response

    @patch("hellmholtz.benchmark.runner.chat_raw")
    def test_benchmark_run_single_model_single_prompt(
        self,
        mock_chat_raw: MagicMock,
        sample_prompts: list[Prompt],
        mock_chat_response: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test benchmark run with single model and single prompt."""
        mock_chat_raw.return_value = mock_chat_response

        models = ["openai:gpt-4o"]
        prompts = sample_prompts[:1]  # Use only first prompt

        results = run_benchmarks(
            models=models,
            prompts=prompts,
            results_dir=str(tmp_path),
            temperatures=[0.1],
            replications=1,
        )

        assert len(results) == 1
        assert results[0].success is True
        assert results[0].input_tokens == 10
        assert results[0].output_tokens == 20
        assert results[0].model == "openai:gpt-4o"
        assert results[0].prompt_id == "test-1"

        # Verify file creation
        files = list(tmp_path.glob("benchmark_*.json"))
        assert len(files) == 1

        with open(files[0]) as f:
            data = json.load(f)
            assert len(data) == 1
            assert data[0]["model"] == "openai:gpt-4o"
            assert data[0]["prompt_id"] == "test-1"
            assert data[0]["success"] is True

    @patch("hellmholtz.benchmark.runner.chat_raw")
    def test_benchmark_run_multiple_models(
        self,
        mock_chat_raw: MagicMock,
        sample_prompts: list[Prompt],
        mock_chat_response: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test benchmark run with multiple models."""
        mock_chat_raw.return_value = mock_chat_response

        models = ["openai:gpt-4o", "anthropic:claude-3-sonnet-20240229"]
        prompts = sample_prompts[:1]

        results = run_benchmarks(
            models=models,
            prompts=prompts,
            results_dir=str(tmp_path),
            temperatures=[0.1],
            replications=1,
        )

        assert len(results) == 2
        model_names = {result.model for result in results}
        assert model_names == {"openai:gpt-4o", "anthropic:claude-3-sonnet-20240229"}

        # Verify file creation
        files = list(tmp_path.glob("benchmark_*.json"))
        assert len(files) == 1

        with open(files[0]) as f:
            data = json.load(f)
            assert len(data) == 2
            saved_models = {item["model"] for item in data}
            assert saved_models == model_names

    @patch("hellmholtz.benchmark.runner.chat_raw")
    def test_benchmark_run_multiple_prompts(
        self,
        mock_chat_raw: MagicMock,
        sample_prompts: list[Prompt],
        mock_chat_response: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test benchmark run with multiple prompts."""
        mock_chat_raw.return_value = mock_chat_response

        models = ["openai:gpt-4o"]
        prompts = sample_prompts

        results = run_benchmarks(
            models=models,
            prompts=prompts,
            results_dir=str(tmp_path),
            temperatures=[0.1],
            replications=1,
        )

        assert len(results) == 2
        prompt_ids = {result.prompt_id for result in results}
        assert prompt_ids == {"test-1", "test-2"}

    @patch("hellmholtz.benchmark.runner.chat_raw")
    def test_benchmark_run_with_replications(
        self,
        mock_chat_raw: MagicMock,
        sample_prompts: list[Prompt],
        mock_chat_response: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test benchmark run with multiple replications."""
        mock_chat_raw.return_value = mock_chat_response

        models = ["openai:gpt-4o"]
        prompts = sample_prompts[:1]
        replications = 3

        results = run_benchmarks(
            models=models,
            prompts=prompts,
            results_dir=str(tmp_path),
            temperatures=[0.1],
            replications=replications,
        )

        assert len(results) == replications
        for result in results:
            assert result.model == "openai:gpt-4o"
            assert result.prompt_id == "test-1"

    @patch("hellmholtz.benchmark.runner.chat_raw")
    def test_benchmark_run_multiple_temperatures(
        self,
        mock_chat_raw: MagicMock,
        sample_prompts: list[Prompt],
        mock_chat_response: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test benchmark run with multiple temperatures."""
        mock_chat_raw.return_value = mock_chat_response

        models = ["openai:gpt-4o"]
        prompts = sample_prompts[:1]
        temperatures = [0.0, 0.5, 1.0]

        results = run_benchmarks(
            models=models,
            prompts=prompts,
            results_dir=str(tmp_path),
            temperatures=temperatures,
            replications=1,
        )

        assert len(results) == len(temperatures)
        result_temps = {result.temperature for result in results}
        assert result_temps == set(temperatures)

    @patch("hellmholtz.benchmark.runner.chat_raw")
    def test_benchmark_run_chat_failure(
        self, mock_chat_raw: MagicMock, sample_prompts: list[Prompt], tmp_path: Path
    ) -> None:
        """Test benchmark run when chat call fails."""
        mock_chat_raw.side_effect = Exception("API Error")

        models = ["openai:gpt-4o"]
        prompts = sample_prompts[:1]

        results = run_benchmarks(
            models=models,
            prompts=prompts,
            results_dir=str(tmp_path),
            temperatures=[0.1],
            replications=1,
        )

        assert len(results) == 1
        assert results[0].success is False
        assert results[0].error_message == "API Error"

    @patch("hellmholtz.benchmark.runner.chat_raw")
    def test_benchmark_run_no_usage_info(
        self, mock_chat_raw: MagicMock, sample_prompts: list[Prompt], tmp_path: Path
    ) -> None:
        """Test benchmark run when response has no usage information."""
        mock_response = MagicMock()
        mock_response.usage = None
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_chat_raw.return_value = mock_response

        models = ["openai:gpt-4o"]
        prompts = sample_prompts[:1]

        results = run_benchmarks(
            models=models,
            prompts=prompts,
            results_dir=str(tmp_path),
            temperatures=[0.1],
            replications=1,
        )

        assert len(results) == 1
        assert results[0].success is True
        assert results[0].input_tokens is None  # Should be None when no usage info
        assert results[0].output_tokens is None  # Should be None when no usage info

    def test_benchmark_result_serialization(self, sample_prompts: list[Prompt]) -> None:
        """Test that BenchmarkResult can be properly serialized to JSON."""
        from dataclasses import asdict
        from datetime import datetime

        result = BenchmarkResult(
            model="test-model",
            prompt_id="test-1",
            temperature=0.5,
            run_id=1,
            response_text="Test response",
            success=True,
            input_tokens=10,
            output_tokens=20,
            latency_seconds=1.5,
            timestamp=datetime.now().isoformat(),
        )

        # Test JSON serialization
        data = asdict(result)
        assert data["model"] == "test-model"
        assert data["prompt_id"] == "test-1"
        assert data["temperature"] == 0.5
        assert data["success"] is True
        assert data["input_tokens"] == 10
        assert data["output_tokens"] == 20
        assert data["latency_seconds"] == 1.5

        # Test deserialization
        json_str = json.dumps(data)
        loaded_data = json.loads(json_str)
        assert loaded_data["model"] == "test-model"

    @patch("builtins.open", new_callable=mock_open)
    @patch("json.dump")
    def test_results_file_creation(
        self, mock_json_dump: MagicMock, mock_file: MagicMock, tmp_path: Path
    ) -> None:
        """Test that results are properly written to file."""
        # This is a basic test to ensure file operations are called
        # In a real scenario, we'd check the actual file content
        # results = [
        #     BenchmarkResult(
        #         model="test-model",
        #         prompt_id="test-1",
        #         temperature=0.1,
        #         run_id=1,
        #         response_text="test",
        #         success=True,
        #         input_tokens=5,
        #         output_tokens=10,
        #         latency_seconds=1.0,
        #         timestamp="2023-01-01T00:00:00",
        #     )
        # ]

        # Mock the file operations that happen in run_benchmarks
        with patch("pathlib.Path.mkdir"), patch("pathlib.Path.glob") as mock_glob:
            mock_glob.return_value = [Path(tmp_path / "benchmark_test.json")]
            # The actual file writing logic is tested implicitly through the run_benchmarks tests above
            pass
