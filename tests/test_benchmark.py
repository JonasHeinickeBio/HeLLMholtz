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
    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.glob")
    def test_results_file_creation(
        self, mock_glob: MagicMock, mock_mkdir: MagicMock, mock_json_dump: MagicMock, mock_file: MagicMock
    ) -> None:
        """Test that results are properly written to file."""
        from hellmholtz.benchmark.runner import BenchmarkResult

        results = [
            BenchmarkResult(
                model="test-model",
                prompt_id="test-1",
                temperature=0.1,
                run_id=1,
                response_text="test",
                success=True,
                input_tokens=5,
                output_tokens=10,
                latency_seconds=1.0,
                timestamp="2023-01-01T00:00:00",
            )
        ]

        # Mock the file operations that happen in run_benchmarks
        mock_glob.return_value = [Path("results/benchmark_test.json")]

        # Import and call the function that writes results
        from hellmholtz.benchmark.runner import save_results
        save_results(results, Path("results"), "benchmark_test")

        # Assert that json.dump was called with the expected data
        mock_json_dump.assert_called_once()
        call_args = mock_json_dump.call_args
        saved_data = call_args[0][0]  # First positional argument is the data
        assert len(saved_data) == 1
        assert saved_data[0]["model"] == "test-model"
        assert saved_data[0]["prompt_id"] == "test-1"


class TestBenchmarkPrompts:
    """Test suite for benchmark prompts functionality."""

    def test_get_all_prompts(self) -> None:
        """Test getting all prompts."""
        from hellmholtz.benchmark.prompts import get_all_prompts

        prompts = get_all_prompts()
        assert isinstance(prompts, list)
        assert len(prompts) > 0
        # Check that all prompts have required attributes
        for prompt in prompts:
            assert hasattr(prompt, 'id')
            assert hasattr(prompt, 'category')
            assert hasattr(prompt, 'messages')
            assert hasattr(prompt, 'description')

    def test_get_prompts_by_category(self) -> None:
        """Test getting prompts by category."""
        from hellmholtz.benchmark.prompts import get_prompts_by_category

        # Test with existing category
        reasoning_prompts = get_prompts_by_category("reasoning")
        assert isinstance(reasoning_prompts, list)
        assert len(reasoning_prompts) > 0
        for prompt in reasoning_prompts:
            assert prompt.category == "reasoning"

        # Test with non-existing category
        empty_prompts = get_prompts_by_category("nonexistent")
        assert isinstance(empty_prompts, list)
        assert len(empty_prompts) == 0

    def test_get_prompt_by_id(self) -> None:
        """Test getting a prompt by ID."""
        from hellmholtz.benchmark.prompts import get_prompt_by_id

        # Test with existing ID
        prompt = get_prompt_by_id("reasoning_001")
        assert prompt is not None
        assert prompt.id == "reasoning_001"
        assert prompt.category == "reasoning"

        # Test with non-existing ID
        prompt = get_prompt_by_id("nonexistent_id")
        assert prompt is None

    def test_prompt_categories_exist(self) -> None:
        """Test that prompts have expected categories."""
        from hellmholtz.benchmark.prompts import get_all_prompts

        prompts = get_all_prompts()
        categories = {prompt.category for prompt in prompts}

        # Check that we have some expected categories
        expected_categories = {"reasoning", "coding", "creative", "knowledge"}
        assert len(categories.intersection(expected_categories)) > 0


class TestBenchmarkEvaluator:
    """Test suite for benchmark evaluator functionality."""

    @pytest.fixture
    def sample_results(self) -> list[BenchmarkResult]:
        """Create sample benchmark results for testing."""
        return [
            BenchmarkResult(
                model="openai:gpt-4o",
                prompt_id="test-1",
                latency_seconds=1.5,
                success=True,
                timestamp="2024-01-01T12:00:00",
                response_text="This is a good response.",
            ),
            BenchmarkResult(
                model="openai:gpt-4o",
                prompt_id="test-2",
                latency_seconds=2.0,
                success=True,
                timestamp="2024-01-01T12:00:00",
                response_text="Another excellent response.",
            ),
            BenchmarkResult(
                model="openai:gpt-4o",
                prompt_id="test-3",
                latency_seconds=1.8,
                success=False,  # Failed result
                timestamp="2024-01-01T12:00:00",
                response_text=None,
            ),
        ]

    @pytest.fixture
    def sample_prompts(self) -> list[Prompt]:
        """Create sample prompts for testing."""
        return [
            Prompt(
                id="test-1",
                category="test",
                messages=[Message(role="user", content="What is 2+2?")],
            ),
            Prompt(
                id="test-2",
                category="test",
                messages=[Message(role="user", content="Explain quantum physics.")],
            ),
            Prompt(
                id="test-3",
                category="test",
                messages=[Message(role="user", content="Write a poem.")],
            ),
        ]

    @patch("hellmholtz.benchmark.evaluator.chat")
    @patch("hellmholtz.client.check_model_availability", return_value=True)
    def test_evaluate_responses_success(self, mock_check_model_availability: MagicMock, mock_chat: MagicMock, sample_results: list[BenchmarkResult], sample_prompts: list[Prompt]) -> None:
        """Test successful evaluation of responses."""

        # Mock judge response
        mock_chat.return_value = "RATING: 8.5\nCRITIQUE: This is a well-structured and accurate response that demonstrates good understanding."

        from hellmholtz.benchmark.evaluator import evaluate_responses

        results = evaluate_responses(sample_results, "openai:gpt-4o", sample_prompts)

        # Check that successful results got ratings and critiques
        assert results[0].rating == 8.5
        assert "well-structured" in results[0].critique
        assert results[1].rating == 8.5
        assert "well-structured" in results[1].critique

        # Failed result should not be evaluated
        assert results[2].rating is None
        assert results[2].critique is None

        # Check that chat was called for successful results only
        assert mock_chat.call_count == 2

    @patch("hellmholtz.benchmark.evaluator.chat")
    def test_evaluate_responses_parsing_errors(self, mock_chat: MagicMock, sample_results: list[BenchmarkResult], sample_prompts: list[Prompt]) -> None:
        """Test evaluation with malformed judge responses."""
        # Mock judge response with missing rating
        mock_chat.return_value = "This response is okay but lacks detail."

        from hellmholtz.benchmark.evaluator import evaluate_responses

        results = evaluate_responses(sample_results, "openai:gpt-4o", sample_prompts)

        # Should handle missing rating gracefully
        assert results[0].rating is None
        assert results[0].critique is None  # No CRITIQUE section

    @patch("hellmholtz.benchmark.evaluator.chat")
    def test_evaluate_responses_chat_error(self, mock_chat: MagicMock, sample_results: list[BenchmarkResult], sample_prompts: list[Prompt]) -> None:
        """Test evaluation when chat call fails."""
        mock_chat.side_effect = Exception("API Error")

        from hellmholtz.benchmark.evaluator import evaluate_responses

        results = evaluate_responses(sample_results, "openai:gpt-4o", sample_prompts)

        # Results should remain unchanged on error
        assert results[0].rating is None
        assert results[0].critique is None

    def test_evaluate_responses_no_matching_prompt(self, sample_results: list[BenchmarkResult]) -> None:
        """Test evaluation when prompt ID doesn't match."""
        from hellmholtz.benchmark.evaluator import evaluate_responses

        # Prompt with different ID
        prompts = [
            Prompt(
                id="different-id",
                category="test",
                messages=[Message(role="user", content="Different prompt")],
            )
        ]

        results = evaluate_responses(sample_results, "openai:gpt-4o", prompts)

        # Should skip evaluation due to no matching prompt
        assert results[0].rating is None
        assert results[0].critique is None

    def test_evaluate_responses_empty_results(self) -> None:
        """Test evaluation with empty results list."""
        from hellmholtz.benchmark.evaluator import evaluate_responses

        results = evaluate_responses([], "openai:gpt-4o", [])

        assert results == []

    @patch("hellmholtz.benchmark.evaluator.chat")
    @patch("hellmholtz.client.check_model_availability", return_value=True)
    def test_evaluate_responses_partial_ratings(self, mock_check_model_availability: MagicMock, mock_chat: MagicMock, sample_results: list[BenchmarkResult], sample_prompts: list[Prompt]) -> None:
        """Test evaluation with only rating but no critique."""
        mock_chat.return_value = "RATING: 7"

        from hellmholtz.benchmark.evaluator import evaluate_responses

        results = evaluate_responses(sample_results, "openai:gpt-4o", sample_prompts)

        assert results[0].rating == 7.0
        assert results[0].critique is None
