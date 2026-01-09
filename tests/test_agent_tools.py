"""
Tests for agent tools.

This module contains tests for the tool system including
the base Tool interface and all concrete tool implementations.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hellmholtz.agent.tools import (
    CalculatorTool,
    FileIOTool,
    SearchTool,
    Tool,
    ToolResult,
)


class TestToolResult:
    """Test suite for ToolResult dataclass."""

    def test_tool_result_success(self) -> None:
        """Test successful tool result."""
        result = ToolResult(success=True, output="Success!", error=None)
        assert result.success is True
        assert result.output == "Success!"
        assert result.error is None
        assert str(result) == "Success!"

    def test_tool_result_failure(self) -> None:
        """Test failed tool result."""
        result = ToolResult(success=False, output="", error="Something went wrong")
        assert result.success is False
        assert result.error == "Something went wrong"
        assert "Error:" in str(result)


class TestCalculatorTool:
    """Test suite for CalculatorTool."""

    @pytest.fixture
    def calculator(self) -> CalculatorTool:
        """Create calculator tool instance."""
        return CalculatorTool()

    def test_calculator_name_and_description(self, calculator: CalculatorTool) -> None:
        """Test calculator tool name and description."""
        assert calculator.name == "calculator"
        assert "mathematical" in calculator.description.lower()

    def test_simple_addition(self, calculator: CalculatorTool) -> None:
        """Test simple addition."""
        result = calculator.execute("2 + 2")
        assert result.success is True
        assert "4" in result.output

    def test_complex_expression(self, calculator: CalculatorTool) -> None:
        """Test complex mathematical expression."""
        result = calculator.execute("(10 * 5) / 2 + 3")
        assert result.success is True
        assert "28" in result.output

    def test_power_operation(self, calculator: CalculatorTool) -> None:
        """Test power operation."""
        result = calculator.execute("2 ** 8")
        assert result.success is True
        assert "256" in result.output

    def test_modulo_operation(self, calculator: CalculatorTool) -> None:
        """Test modulo operation."""
        result = calculator.execute("10 % 3")
        assert result.success is True
        assert "1" in result.output

    def test_division_by_zero(self, calculator: CalculatorTool) -> None:
        """Test division by zero error handling."""
        result = calculator.execute("10 / 0")
        assert result.success is False
        assert "division by zero" in result.error.lower()

    def test_invalid_expression(self, calculator: CalculatorTool) -> None:
        """Test invalid expression error handling."""
        result = calculator.execute("not a number")
        assert result.success is False
        assert "invalid" in result.error.lower()

    def test_unsafe_expression(self, calculator: CalculatorTool) -> None:
        """Test that unsafe expressions are rejected."""
        # Try to use non-math operations
        result = calculator.execute("__import__('os').system('ls')")
        assert result.success is False


class TestFileIOTool:
    """Test suite for FileIOTool."""

    @pytest.fixture
    def temp_workspace(self) -> Path:
        """Create temporary workspace directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def file_io(self, temp_workspace: Path) -> FileIOTool:
        """Create FileIOTool with temporary workspace."""
        return FileIOTool(workspace_dir=str(temp_workspace))

    def test_file_io_name_and_description(self, file_io: FileIOTool) -> None:
        """Test file I/O tool name and description."""
        assert file_io.name == "file_io"
        assert "read" in file_io.description.lower()
        assert "write" in file_io.description.lower()

    def test_write_file(self, file_io: FileIOTool, temp_workspace: Path) -> None:
        """Test writing file."""
        test_file = temp_workspace / "test.txt"
        result = file_io.execute(
            operation="write", path=str(test_file), content="Hello, World!"
        )
        assert result.success is True
        assert test_file.exists()
        assert test_file.read_text() == "Hello, World!"

    def test_read_file(self, file_io: FileIOTool, temp_workspace: Path) -> None:
        """Test reading file."""
        test_file = temp_workspace / "test.txt"
        test_file.write_text("Test content")

        result = file_io.execute(operation="read", path=str(test_file))
        assert result.success is True
        assert "Test content" in result.output

    def test_list_directory(self, file_io: FileIOTool, temp_workspace: Path) -> None:
        """Test listing directory."""
        # Create some files
        (temp_workspace / "file1.txt").write_text("content1")
        (temp_workspace / "file2.txt").write_text("content2")
        (temp_workspace / "subdir").mkdir()

        result = file_io.execute(operation="list", path=str(temp_workspace))
        assert result.success is True
        assert "file1.txt" in result.output
        assert "file2.txt" in result.output
        assert "subdir" in result.output

    def test_read_nonexistent_file(self, file_io: FileIOTool, temp_workspace: Path) -> None:
        """Test reading non-existent file."""
        # Use a path within the workspace
        nonexistent_file = temp_workspace / "nonexistent.txt"
        result = file_io.execute(operation="read", path=str(nonexistent_file))
        assert result.success is False
        assert "not found" in result.error.lower()

    def test_invalid_operation(self, file_io: FileIOTool) -> None:
        """Test invalid operation."""
        result = file_io.execute(operation="delete", path="test.txt")
        assert result.success is False
        assert "unknown operation" in result.error.lower()

    def test_write_without_content(self, file_io: FileIOTool) -> None:
        """Test write operation without content."""
        result = file_io.execute(operation="write", path="test.txt")
        assert result.success is False
        assert "content required" in result.error.lower()

    def test_workspace_restriction(self, temp_workspace: Path) -> None:
        """Test that file operations are restricted to workspace."""
        file_io = FileIOTool(workspace_dir=str(temp_workspace))

        # Try to access file outside workspace
        result = file_io.execute(operation="read", path="/etc/passwd")
        assert result.success is False
        assert "outside workspace" in result.error.lower()


class TestSearchTool:
    """Test suite for SearchTool."""

    @pytest.fixture
    def search_tool(self) -> SearchTool:
        """Create search tool instance."""
        return SearchTool(max_results=3)

    def test_search_name_and_description(self, search_tool: SearchTool) -> None:
        """Test search tool name and description."""
        assert search_tool.name == "search"
        assert "search" in search_tool.description.lower()
        assert "duckduckgo" in search_tool.description.lower()

    def test_empty_query(self, search_tool: SearchTool) -> None:
        """Test search with empty query."""
        result = search_tool.execute("")
        assert result.success is False
        assert "empty" in result.error.lower()

    @pytest.mark.network
    @pytest.mark.skipif(
        True, reason="Skipping network tests by default - duckduckgo-search may not be installed"
    )
    def test_search_basic(self, search_tool: SearchTool) -> None:
        """Test basic search functionality (requires network and duckduckgo-search)."""
        result = search_tool.execute("Python programming language")
        # Will fail if duckduckgo-search not installed, which is expected
        if result.success:
            assert "python" in result.output.lower()

    def test_search_without_library(self) -> None:
        """Test search behavior when library is not available."""
        # Create tool and check if library is available
        tool = SearchTool()

        # Mock the library check
        with patch.object(tool, "_ddgs_available", False):
            result = tool.execute("test query")
            assert result.success is False
            assert "duckduckgo-search" in result.error.lower()


class TestToolInterface:
    """Test suite for Tool base interface."""

    def test_tool_is_abstract(self) -> None:
        """Test that Tool cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Tool()  # type: ignore

    def test_concrete_tools_implement_interface(self) -> None:
        """Test that all concrete tools properly implement Tool interface."""
        tools = [CalculatorTool(), FileIOTool(), SearchTool()]

        for tool in tools:
            assert isinstance(tool, Tool)
            assert hasattr(tool, "name")
            assert hasattr(tool, "description")
            assert hasattr(tool, "execute")
            assert callable(tool.execute)
            assert isinstance(tool.name, str)
            assert isinstance(tool.description, str)
            assert len(tool.name) > 0
            assert len(tool.description) > 0

    def test_tool_string_representation(self) -> None:
        """Test __str__ method of tools."""
        calculator = CalculatorTool()
        str_repr = str(calculator)
        assert calculator.name in str_repr
        assert calculator.description in str_repr
