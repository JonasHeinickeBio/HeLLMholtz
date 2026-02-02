"""Base tool interface for agent system."""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """Result from tool execution."""

    success: bool = Field(..., description="Whether tool execution succeeded")
    output: str = Field(..., description="Tool output or error message")
    error: str | None = Field(None, description="Error details if execution failed")

    def __str__(self) -> str:
        """String representation of tool result."""
        if self.success:
            return self.output
        return f"Error: {self.error or self.output}"


class Tool(ABC):
    """Abstract base class for agent tools.

    All tools must implement the execute method and provide
    name and description properties for the agent to use them.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name used in agent commands."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what the tool does and how to use it."""
        pass

    @abstractmethod
    def execute(self, *args: Any, **kwargs: Any) -> ToolResult:
        """Execute the tool with given arguments.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            ToolResult with execution outcome
        """
        pass

    def __str__(self) -> str:
        """String representation for agent prompts."""
        return f"{self.name}: {self.description}"
