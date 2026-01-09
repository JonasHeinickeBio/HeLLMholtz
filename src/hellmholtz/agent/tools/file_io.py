"""File I/O tool for reading and writing files."""

import logging
from pathlib import Path
from typing import Any

from hellmholtz.agent.tools.base import Tool, ToolResult

logger = logging.getLogger(__name__)


class FileIOTool(Tool):
    """Tool for safe file reading and writing operations.

    Provides read and write operations with path validation
    to prevent directory traversal attacks.
    """

    def __init__(self, workspace_dir: str | None = None) -> None:
        """Initialize file I/O tool.

        Args:
            workspace_dir: Optional workspace directory to restrict operations.
                         If provided, all file operations are restricted to this directory.
        """
        self.workspace_dir = Path(workspace_dir) if workspace_dir else None
        if self.workspace_dir:
            self.workspace_dir = self.workspace_dir.resolve()
            self.workspace_dir.mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str:
        """Tool name."""
        return "file_io"

    @property
    def description(self) -> str:
        """Tool description."""
        workspace_info = (
            f" (restricted to {self.workspace_dir})" if self.workspace_dir else ""
        )
        return (
            f"Reads and writes files{workspace_info}. "
            "Operations: 'read' to read file content, 'write' to write content to file, "
            "'list' to list files in directory. "
            "Args: operation='read|write|list', path='file_path', content='...' (for write)"
        )

    def _validate_path(self, path: str) -> Path:
        """Validate and resolve file path.

        Args:
            path: File path to validate

        Returns:
            Resolved Path object

        Raises:
            ValueError: If path is invalid or outside workspace
        """
        file_path = Path(path).resolve()

        # If workspace is set, ensure path is within workspace
        if self.workspace_dir:
            try:
                file_path.relative_to(self.workspace_dir)
            except ValueError:
                raise ValueError(
                    f"Path {path} is outside workspace directory {self.workspace_dir}"
                )

        return file_path

    def execute(
        self, operation: str, path: str, content: str | None = None, **kwargs: Any
    ) -> ToolResult:
        """Execute file I/O operation.

        Args:
            operation: Operation to perform ('read', 'write', 'list')
            path: File or directory path
            content: Content to write (required for 'write' operation)
            **kwargs: Additional arguments (ignored)

        Returns:
            ToolResult with operation result or error
        """
        try:
            if operation == "read":
                return self._read_file(path)
            elif operation == "write":
                if content is None:
                    return ToolResult(
                        success=False, output="", error="Content required for write operation"
                    )
                return self._write_file(path, content)
            elif operation == "list":
                return self._list_directory(path)
            else:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Unknown operation: {operation}. Use 'read', 'write', or 'list'",
                )
        except ValueError as e:
            return ToolResult(success=False, output="", error=str(e))
        except Exception as e:
            logger.error(f"File I/O error: {e}")
            return ToolResult(success=False, output="", error=f"File operation failed: {str(e)}")

    def _read_file(self, path: str) -> ToolResult:
        """Read file content."""
        try:
            file_path = self._validate_path(path)
            if not file_path.exists():
                return ToolResult(success=False, output="", error=f"File not found: {path}")
            if not file_path.is_file():
                return ToolResult(success=False, output="", error=f"Not a file: {path}")

            content = file_path.read_text(encoding="utf-8")
            return ToolResult(
                success=True,
                output=f"Content of {path}:\n{content}",
                error=None,
            )
        except UnicodeDecodeError:
            return ToolResult(
                success=False, output="", error=f"Cannot read binary file: {path}"
            )

    def _write_file(self, path: str, content: str) -> ToolResult:
        """Write content to file."""
        file_path = self._validate_path(path)

        # Create parent directory if needed
        file_path.parent.mkdir(parents=True, exist_ok=True)

        file_path.write_text(content, encoding="utf-8")
        return ToolResult(
            success=True,
            output=f"Successfully wrote {len(content)} characters to {path}",
            error=None,
        )

    def _list_directory(self, path: str) -> ToolResult:
        """List directory contents."""
        dir_path = self._validate_path(path)

        if not dir_path.exists():
            return ToolResult(success=False, output="", error=f"Directory not found: {path}")
        if not dir_path.is_dir():
            return ToolResult(success=False, output="", error=f"Not a directory: {path}")

        entries = []
        for entry in sorted(dir_path.iterdir()):
            entry_type = "DIR" if entry.is_dir() else "FILE"
            size = entry.stat().st_size if entry.is_file() else "-"
            entries.append(f"  [{entry_type}] {entry.name} ({size} bytes)")

        output = f"Contents of {path}:\n" + "\n".join(entries)
        return ToolResult(success=True, output=output, error=None)
