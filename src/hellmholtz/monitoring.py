"""
Model Availability Monitoring for Blablador Service

This module provides functionality to monitor the availability of models
in the Blablador API and compare them with the configured models.
"""

import os
from pathlib import Path
import time
from typing import Any, cast

import requests

from hellmholtz.client import chat
from hellmholtz.providers.blablador_config import KNOWN_MODELS, BlabladorModel


class ModelAvailabilityMonitor:
    """Monitor for Blablador model availability and configuration consistency."""

    def __init__(self, api_key: str | None = None, api_base: str | None = None):
        """Initialize the monitor with API credentials.

        Args:
            api_key: Blablador API key. If None, uses BLABLADOR_API_KEY env var.
            api_base: API base URL. If None, uses BLABLADOR_API_BASE env var.
        """
        self.api_key = api_key or os.getenv("BLABLADOR_API_KEY")
        self.api_base = api_base or os.getenv(
            "BLABLADOR_API_BASE", "https://api.blablador.example.com/v1"
        )

        if not self.api_key:
            raise ValueError(
                "BLABLADOR_API_KEY not found. Set the environment variable "
                "or pass api_key parameter."
            )

        self.headers = {"Authorization": f"Bearer {self.api_key}"}
        self.test_message = [{"role": "user", "content": "Hello"}]

    def get_api_models(self) -> list[dict[str, Any]]:
        """Fetch current models from the Blablador API.

        Returns:
            List of model dictionaries from the API.

        Raises:
            requests.RequestException: If API request fails.
        """
        try:
            response = requests.get(f"{self.api_base}/models", headers=self.headers, timeout=30)
            response.raise_for_status()
            data = response.json().get("data", [])
            return cast(list[dict[str, Any]], data)
        except Exception as e:
            raise RuntimeError(f"Failed to fetch models from API: {e}") from e

    def get_configured_models(self) -> dict[str, BlabladorModel]:
        """Get models configured in blablador_config.py.

        Returns:
            Dictionary mapping API IDs to BlabladorModel instances.
        """
        return {model.api_id: model for model in KNOWN_MODELS}

    def test_model_accessibility(
        self, model_name: str, timeout: float = 10.0
    ) -> tuple[bool, float]:
        """Test if a model is actually accessible by making a test request.

        Args:
            model_name: Name of the model to test.
            timeout: Timeout for the request in seconds.

        Returns:
            Tuple of (is_accessible, latency_seconds).
        """
        try:
            start_time = time.time()
            response = chat(
                f"blablador:{model_name}", self.test_message, max_tokens=5, timeout=timeout
            )
            latency = time.time() - start_time
            return bool(response and response.strip()), latency
        except Exception:
            return False, 0.0

    def analyze_availability(self, test_accessibility: bool = False) -> dict[str, Any]:
        """Analyze model availability and configuration consistency.

        Args:
            test_accessibility: Whether to test actual model accessibility.

        Returns:
            Dictionary containing analysis results.
        """
        print("🔍 Fetching current API models...")
        api_models = self.get_api_models()
        api_model_ids = {model["id"]: model for model in api_models}

        print(f"📋 Found {len(api_models)} models in API")
        print(f"⚙️  Found {len(KNOWN_MODELS)} models in configuration")

        configured_models = self.get_configured_models()

        # Categorize models
        configured_and_available = []
        configured_not_available = []
        available_not_configured = []
        accessibility_results = {}

        # Check configured models
        print("\n🔎 Checking configured models...")
        for api_id, config_model in configured_models.items():
            if api_id in api_model_ids:
                configured_and_available.append((api_id, config_model))
                if test_accessibility:
                    print(f"  🧪 Testing {config_model.name}...")
                    accessible, latency = self.test_model_accessibility(config_model.name)
                    accessibility_results[config_model.name] = {
                        "accessible": accessible,
                        "latency": latency,
                        "api_id": api_id,
                    }
            else:
                configured_not_available.append((api_id, config_model))

        # Check API models not in configuration
        for api_id, api_model in api_model_ids.items():
            if api_id not in configured_models:
                available_not_configured.append((api_id, api_model))

        return {
            "api_models_count": len(api_models),
            "configured_models_count": len(KNOWN_MODELS),
            "configured_and_available": configured_and_available,
            "configured_not_available": configured_not_available,
            "available_not_configured": available_not_configured,
            "accessibility_results": accessibility_results,
            "timestamp": time.time(),
        }

    def generate_report(self, analysis: dict[str, Any], test_accessibility: bool = False) -> str:
        """Generate a comprehensive availability report."""
        lines = []
        lines.extend(self._generate_report_header(analysis))
        lines.extend(self._generate_report_summary(analysis))
        lines.extend(self._generate_report_sections(analysis, test_accessibility))
        lines.extend(self._generate_report_recommendations(analysis))
        return "\n".join(lines)

    def _generate_report_header(self, analysis: dict[str, Any]) -> list[str]:
        """Generate the report header."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(analysis["timestamp"]))
        return [
            "📊 Blablador Model Availability Report",
            "=" * 50,
            f"Generated: {timestamp}",
            "",
        ]

    def _generate_report_summary(self, analysis: dict[str, Any]) -> list[str]:
        """Generate the report summary section."""
        return [
            "📈 Summary:",
            f"  • API Models: {analysis['api_models_count']}",
            f"  • Configured Models: {analysis['configured_models_count']}",
            f"  • Available & Configured: {len(analysis['configured_and_available'])}",
            f"  • Configured but Unavailable: {len(analysis['configured_not_available'])}",
            f"  • Available but Unconfigured: {len(analysis['available_not_configured'])}",
            "",
        ]

    def _generate_report_sections(
        self, analysis: dict[str, Any], test_accessibility: bool
    ) -> list[str]:
        """Generate the detailed sections of the report."""
        lines = []

        # Available and configured
        if analysis["configured_and_available"]:
            lines.append("✅ Available & Configured Models:")
            for _api_id, model in analysis["configured_and_available"]:
                status = ""
                if test_accessibility and model.name in analysis["accessibility_results"]:
                    result = analysis["accessibility_results"][model.name]
                    if result["accessible"]:
                        status = f" (✅ Accessible, {result['latency']:.2f}s)"
                    else:
                        status = " (❌ Not accessible)"
                lines.append(f"  • {model.name}{status}")
            lines.append("")

        # Configured but not available
        if analysis["configured_not_available"]:
            lines.append("⚠️  Configured but Not Available:")
            for api_id, model in analysis["configured_not_available"]:
                lines.append(f"  • {model.name} (API ID: {api_id})")
            lines.append("")

        # Available but not configured
        if analysis["available_not_configured"]:
            lines.append("🔍 Available but Not Configured:")
            for _api_id, api_model in analysis["available_not_configured"]:
                model_id = api_model.get("id", "unknown")
                model_obj = api_model.get("object", "unknown")
                lines.append(f"  • {model_id} - {model_obj}")
            lines.append("")

        return lines

    def _generate_report_recommendations(self, analysis: dict[str, Any]) -> list[str]:
        """Generate the recommendations section."""
        lines = ["💡 Recommendations:"]

        if analysis["configured_not_available"]:
            lines.extend(
                [
                    "  • Review configured models that are no longer available",
                    "  • Consider removing or updating these configurations",
                ]
            )

        if analysis["available_not_configured"]:
            lines.extend(
                [
                    "  • Consider adding newly available models to configuration",
                    "  • Test new models for compatibility and performance",
                ]
            )

        if not analysis["configured_not_available"] and not analysis["available_not_configured"]:
            lines.append("  • Configuration is up-to-date with API ✅")

        return lines

    def save_report(self, report: str, filename: str | None = None) -> str:
        """Save the report to a file.

        Args:
            report: Report content to save.
            filename: Optional filename. If None, generates timestamped filename.

        Returns:
            Path to the saved file.
        """
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"model_availability_report_{timestamp}.txt"

        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        filepath = reports_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report)

        return str(filepath)


def monitor_models(
    test_accessibility: bool = False,
    save_report: bool = True,
    api_key: str | None = None,
    api_base: str | None = None,
) -> str:
    """Convenience function to monitor model availability.

    Args:
        test_accessibility: Whether to test actual model accessibility.
        save_report: Whether to save the report to file.
        api_key: Optional API key override.
        api_base: Optional API base URL override.

    Returns:
        The generated report content.
    """
    monitor = ModelAvailabilityMonitor(api_key=api_key, api_base=api_base)
    analysis = monitor.analyze_availability(test_accessibility=test_accessibility)
    report = monitor.generate_report(analysis, test_accessibility=test_accessibility)

    if save_report:
        filepath = monitor.save_report(report)
        print(f"\n💾 Report saved to: {filepath}")

    return report
