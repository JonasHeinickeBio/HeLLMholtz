"""
Main reporting module for HeLLMholtz benchmark results.
"""

from .html import (
    generate_html_report,
    generate_html_report_detailed,
    generate_html_report_full,
    generate_html_report_simple,
)
from .markdown import (
    generate_markdown_report,
    summarize_results,
)
from .stats import (
    analyze_performance_trends,
    generate_insights,
)
from .utils import export_to_csv, load_results

# Re-export functions for backward compatibility
__all__ = [
    "generate_markdown_report",
    "generate_html_report",
    "generate_html_report_simple",
    "generate_html_report_detailed",
    "generate_html_report_full",
    "summarize_results",
    "export_to_csv",
    "load_results",
    "analyze_performance_trends",
    "generate_insights",
]
