"""Tests for the HeLLMholtz prompt loading functionality."""

import json
import pytest
import tempfile
from pathlib import Path

from hellmholtz.core.prompts import load_prompts, Prompt, Message


class TestPromptLoading:
    """Test suite for prompt loading functionality."""

    @pytest.fixture
    def sample_json_prompts(self) -> str:
        """Sample JSON prompts data."""
        prompts_data = [
            {
                "id": "test-1",
                "category": "reasoning",
                "description": "Test reasoning prompt",
                "messages": [
                    {"role": "user", "content": "What is 2+2?"}
                ],
                "expected_output": "4"
            },
            {
                "id": "test-2",
                "category": "coding",
                "description": "Test coding prompt",
                "messages": [
                    {"role": "system", "content": "You are a Python expert."},
                    {"role": "user", "content": "Write a function to reverse a string."}
                ]
            }
        ]
        return json.dumps(prompts_data)

    @pytest.fixture
    def sample_text_prompts(self) -> str:
        """Sample text prompts data."""
        return "What is the capital of France?\nExplain quantum computing.\nWrite a Python hello world program."

    def test_load_json_prompts(self, sample_json_prompts: str) -> None:
        """Test loading prompts from JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write(sample_json_prompts)
            json_file = f.name

        try:
            prompts = load_prompts(json_file)

            assert len(prompts) == 2

            # Check first prompt
            prompt1 = prompts[0]
            assert isinstance(prompt1, Prompt)
            assert prompt1.id == "test-1"
            assert prompt1.category == "reasoning"
            assert prompt1.description == "Test reasoning prompt"
            assert len(prompt1.messages) == 1
            assert prompt1.messages[0].role == "user"
            assert prompt1.messages[0].content == "What is 2+2?"
            assert prompt1.expected_output == "4"

            # Check second prompt
            prompt2 = prompts[1]
            assert prompt2.id == "test-2"
            assert prompt2.category == "coding"
            assert len(prompt2.messages) == 2
            assert prompt2.messages[0].role == "system"
            assert prompt2.messages[1].role == "user"

        finally:
            Path(json_file).unlink()

    def test_load_text_prompts(self, sample_text_prompts: str) -> None:
        """Test loading prompts from text file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(sample_text_prompts)
            text_file = f.name

        try:
            prompts = load_prompts(text_file)

            assert len(prompts) == 3

            # Check prompts are created correctly
            for i, prompt in enumerate(prompts):
                assert isinstance(prompt, Prompt)
                assert prompt.id == f"prompt_{i:03d}"
                assert prompt.category == "custom"
                assert len(prompt.messages) == 1
                assert prompt.messages[0].role == "user"

            # Check content
            assert "capital of France" in prompts[0].messages[0].content
            assert "quantum computing" in prompts[1].messages[0].content
            assert "Python hello world" in prompts[2].messages[0].content

        finally:
            Path(text_file).unlink()

    def test_load_prompts_with_category_filter(self, sample_json_prompts: str) -> None:
        """Test loading prompts with category filtering."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write(sample_json_prompts)
            json_file = f.name

        try:
            # Filter by reasoning category
            prompts = load_prompts(json_file, category="reasoning")
            assert len(prompts) == 1
            assert prompts[0].category == "reasoning"

            # Filter by coding category
            prompts = load_prompts(json_file, category="coding")
            assert len(prompts) == 1
            assert prompts[0].category == "coding"

            # Filter by non-existent category
            prompts = load_prompts(json_file, category="nonexistent")
            assert len(prompts) == 0

        finally:
            Path(json_file).unlink()

    def test_load_prompts_file_not_found(self) -> None:
        """Test error handling when prompt file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_prompts("nonexistent_file.json")

    def test_load_prompts_invalid_json(self) -> None:
        """Test error handling for invalid JSON."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content {")
            json_file = f.name

        try:
            with pytest.raises(ValueError, match="Invalid JSON"):
                load_prompts(json_file)
        finally:
            Path(json_file).unlink()

    def test_load_prompts_empty_file(self) -> None:
        """Test loading from empty file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            # Empty file
            text_file = f.name

        try:
            prompts = load_prompts(text_file)
            assert len(prompts) == 0
        finally:
            Path(text_file).unlink()

    def test_load_prompts_empty_json_array(self) -> None:
        """Test loading from empty JSON array."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("[]")
            json_file = f.name

        try:
            prompts = load_prompts(json_file)
            assert len(prompts) == 0
        finally:
            Path(json_file).unlink()

    def test_load_prompts_with_category_filter(self) -> None:
        """Test loading prompts with category filtering."""
        prompts_data = [
            {
                "id": "reasoning-1",
                "category": "reasoning",
                "messages": [{"role": "user", "content": "Test reasoning"}]
            },
            {
                "id": "coding-1",
                "category": "coding",
                "messages": [{"role": "user", "content": "Test coding"}]
            },
            {
                "id": "reasoning-2",
                "category": "reasoning",
                "messages": [{"role": "user", "content": "Another reasoning"}]
            }
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(prompts_data, f)
            json_file = f.name

        try:
            # Load all prompts
            all_prompts = load_prompts(json_file)
            assert len(all_prompts) == 3

            # Load only reasoning prompts
            reasoning_prompts = load_prompts(json_file, category="reasoning")
            assert len(reasoning_prompts) == 2
            assert all(p.category == "reasoning" for p in reasoning_prompts)

            # Load only coding prompts
            coding_prompts = load_prompts(json_file, category="coding")
            assert len(coding_prompts) == 1
            assert coding_prompts[0].category == "coding"

            # Load non-existent category
            empty_prompts = load_prompts(json_file, category="nonexistent")
            assert len(empty_prompts) == 0

        finally:
            Path(json_file).unlink()

    def test_load_prompts_file_format_override(self) -> None:
        """Test loading prompts with explicit file format override."""
        # Create a .custom file with JSON content
        json_content = [
            {
                "id": "test-1",
                "category": "test",
                "messages": [{"role": "user", "content": "Test message"}]
            }
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.custom', delete=False) as f:
            json.dump(json_content, f)
            custom_file = f.name

        try:
            # Should fail without format override (unsupported extension)
            with pytest.raises(ValueError, match="Unsupported file extension"):
                load_prompts(custom_file)

            # Should work with format override
            prompts = load_prompts(custom_file, file_format="json")
            assert len(prompts) == 1
            assert prompts[0].id == "test-1"

        finally:
            Path(custom_file).unlink()
