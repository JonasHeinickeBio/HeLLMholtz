"""Web search tool using DuckDuckGo."""

import logging
from typing import Any

from hellmholtz.agent.tools.base import Tool, ToolResult

logger = logging.getLogger(__name__)


class SearchTool(Tool):
    """Tool for web search using DuckDuckGo.

    Uses the duckduckgo-search library which doesn't require API keys.
    Falls back to a simple implementation if the library is not available.
    """

    def __init__(self, max_results: int = 5) -> None:
        """Initialize search tool.

        Args:
            max_results: Maximum number of search results to return
        """
        self.max_results = max_results
        self._ddgs_available = self._check_ddgs()

    def _check_ddgs(self) -> bool:
        """Check if duckduckgo-search library is available."""
        try:
            import duckduckgo_search  # noqa: F401

            return True
        except ImportError:
            logger.warning(
                "duckduckgo-search not installed. Search functionality will be limited. "
                "Install with: pip install duckduckgo-search"
            )
            return False

    @property
    def name(self) -> str:
        """Tool name."""
        return "search"

    @property
    def description(self) -> str:
        """Tool description."""
        return (
            "Searches the web using DuckDuckGo. "
            f"Input: A search query string. Returns up to {self.max_results} results "
            "with titles and snippets."
        )

    def execute(self, query: str, **kwargs: Any) -> ToolResult:
        """Execute web search.

        Args:
            query: Search query string
            **kwargs: Additional arguments (ignored)

        Returns:
            ToolResult with search results or error
        """
        if not query or not query.strip():
            return ToolResult(
                success=False, output="", error="Search query cannot be empty"
            )

        if not self._ddgs_available:
            return ToolResult(
                success=False,
                output="",
                error=(
                    "Search functionality requires duckduckgo-search library. "
                    "Install with: pip install duckduckgo-search"
                ),
            )

        try:
            from duckduckgo_search import DDGS

            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=self.max_results))

            if not results:
                return ToolResult(
                    success=True, output="No results found for the query.", error=None
                )

            # Format results
            output_lines = [f"Search results for '{query}':\n"]
            for i, result in enumerate(results, 1):
                title = result.get("title", "No title")
                snippet = result.get("body", "No description")
                link = result.get("href", "")
                output_lines.append(f"{i}. {title}")
                output_lines.append(f"   {snippet}")
                if link:
                    output_lines.append(f"   URL: {link}")
                output_lines.append("")

            return ToolResult(
                success=True, output="\n".join(output_lines), error=None
            )

        except Exception as e:
            logger.error(f"Search error: {e}")
            return ToolResult(
                success=False, output="", error=f"Search failed: {str(e)}"
            )
