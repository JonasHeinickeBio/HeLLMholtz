"""
Model Availability Monitoring for Blablador Service

This module provides functionality to monitor the availability of models
in the Blablador API and compare them with the configured models.
"""

from datetime import datetime
import os
from pathlib import Path
import time
from typing import Any, cast

import requests
import yaml

from hellmholtz.client import chat
from hellmholtz.providers.blablador import parse_api_model_ids
from hellmholtz.providers.blablador_config import (
    DEFAULT_TOKEN_LIMIT,
    KNOWN_MODELS,
    BlabladorModel,
    _get_online_token_limit,
    get_model_by_name,
)


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
        raw_model_ids = [model.get("id") for model in api_models if isinstance(model, dict)]
        parsed_api_models = parse_api_model_ids(
            [model_id for model_id in raw_model_ids if isinstance(model_id, str)]
        )
        api_model_ids = {model.api_id: model for model in parsed_api_models}
        api_models_by_api_id: dict[str, dict[str, Any]] = {}
        for raw_api_model, parsed_api_model in zip(api_models, parsed_api_models, strict=False):
            api_models_by_api_id.setdefault(parsed_api_model.api_id, raw_api_model)

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
        for api_id in api_model_ids:
            if api_id not in configured_models:
                api_model = api_models_by_api_id.get(api_id, {"id": api_id, "object": "model"})
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

    def load_model_status(self) -> dict[str, Any]:
        """Load model status from YAML file.

        Returns:
            Dictionary containing model status data.
        """
        yaml_path = Path("models_status.yaml")
        if not yaml_path.exists():
            return {}

        try:
            with open(yaml_path, encoding="utf-8") as f:
                data: Any = yaml.safe_load(f) or {}
                # Handle nested structure with 'models' key
                result = data.get("models", data) if isinstance(data, dict) else {}
                return result if isinstance(result, dict) else {}
        except Exception as e:
            print(f"⚠️  Warning: Could not load model status from YAML: {e}")
            return {}

    def save_model_status(self, status_data: dict[str, Any]) -> None:
        """Save model status to YAML file.

        Args:
            status_data: Model status data to save.
        """
        yaml_path = Path("models_status.yaml")

        # Load existing data to preserve structure
        existing_data: dict[str, Any] = {}
        if yaml_path.exists():
            try:
                with open(yaml_path, encoding="utf-8") as f:
                    existing_data = yaml.safe_load(f) or {}
            except Exception:
                existing_data = {}

        # Update the models section
        if "models" in existing_data:
            existing_data["models"] = status_data
        else:
            existing_data = {"models": status_data}

        # Add/update metadata
        existing_data["# This file is automatically updated by the model availability checker"] = (
            None
        )
        existing_data["# Last updated"] = time.strftime("%Y-%m-%d")
        existing_data["# Total models"] = len(status_data)
        available_count = sum(1 for data in status_data.values() if data.get("available", False))
        existing_data["# Working"] = available_count
        existing_data["# Broken"] = len(status_data) - available_count

        try:
            with open(yaml_path, "w", encoding="utf-8") as f:
                yaml.dump(existing_data, f, default_flow_style=False, sort_keys=False)
            print(f"💾 Model status saved to: {yaml_path}")
        except Exception as e:
            print(f"❌ Error saving model status to YAML: {e}")

    def check_all_models_automatically(self) -> dict[str, Any]:
        """Automatically check all models and update YAML status.

        Returns:
            Updated model status dictionary.
        """
        print("🔄 Starting automatic model availability check...")

        # Load existing status
        status_data = self.load_model_status()

        # Get current API models
        try:
            api_models = self.get_api_models()
            api_model_ids = {model["id"]: model for model in api_models}
        except Exception as e:
            print(f"❌ Failed to fetch API models: {e}")
            return status_data

        # Update status for all models
        updated_count = 0
        for model in KNOWN_MODELS:
            model_key = model.name
            api_id = model.api_id

            # Initialize or get existing entry
            if model_key not in status_data:
                status_data[model_key] = {
                    "name": model.name,
                    "description": model.description,
                    "tokens": model.max_context_tokens,
                    "available": False,
                    "latency": None,
                    "category": self._categorize_model(model.name),
                    "provider": "blablador",
                    "api_id": api_id,
                    "last_checked": None,
                }

            # Check availability
            is_available = api_id in api_model_ids
            status_data[model_key]["available"] = is_available

            # Test accessibility if available
            now_ts = time.time()
            now_dt = datetime.fromtimestamp(now_ts).isoformat(sep=" ", timespec="seconds")
            if is_available:
                print(f"  🧪 Testing {model.name}...")
                accessible, latency = self.test_model_accessibility(model.name)
                status_data[model_key]["latency"] = round(latency, 3) if accessible else None
                status_data[model_key]["last_checked"] = now_ts
                status_data[model_key]["last_checked_datetime"] = now_dt
                updated_count += 1
            else:
                status_data[model_key]["latency"] = None
                status_data[model_key]["last_checked"] = now_ts
                status_data[model_key]["last_checked_datetime"] = now_dt

        # Save updated status
        self.save_model_status(status_data)

        print(f"✅ Checked {updated_count} models, status updated in YAML")
        return status_data

    def _categorize_model(self, model_name: str) -> str:
        """Categorize a model based on its name.

        Args:
            model_name: Name of the model.

        Returns:
            Category string.
        """
        name_lower = model_name.lower()

        if "alias" in name_lower:
            return "alias"
        elif "legacy" in name_lower or "old" in name_lower:
            return "legacy"
        elif any(term in name_lower for term in ["3b", "7b", "14b", "32b", "70b", "120b", "405b"]):
            return "base_model"
        elif any(term in name_lower for term in ["instruct", "chat", "conversational"]):
            return "instruction_tuned"
        else:
            return "other"

    def generate_enhanced_report(self, include_yaml_status: bool = True) -> str:
        """Generate an enhanced report including YAML status data.

        Args:
            include_yaml_status: Whether to include data from YAML status file.

        Returns:
            Enhanced report content.
        """
        lines = []
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        lines.extend(
            [
                "📊 Enhanced Blablador Model Status Report",
                "=" * 55,
                f"Generated: {timestamp}",
                "",
            ]
        )

        if include_yaml_status:
            status_data = self.load_model_status()
            if status_data:
                lines.extend(self._generate_yaml_status_section(status_data))
            else:
                lines.append(
                    "⚠️  No YAML status data found. Run check_all_models_automatically() first."
                )
                lines.append("")

        # Generate standard analysis
        try:
            analysis = self.analyze_availability(test_accessibility=True)
            lines.extend(self._generate_report_summary(analysis))
            lines.extend(self._generate_report_sections(analysis, test_accessibility=True))
            lines.extend(self._generate_report_recommendations(analysis))
        except Exception as e:
            lines.append(f"❌ Error generating analysis: {e}")

        return "\n".join(lines)

    def _generate_yaml_status_section(self, status_data: dict[str, Any]) -> list[str]:
        """Generate the YAML status section of the report.

        Args:
            status_data: Model status data from YAML.

        Returns:
            List of report lines.
        """
        lines = ["📋 Model Status from YAML:", ""]

        # Group by category
        categories: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        for model_name, data in status_data.items():
            category = data.get("category", "other")
            if category not in categories:
                categories[category] = []
            categories[category].append((model_name, data))

        # Sort categories
        category_order = ["base_model", "instruction_tuned", "alias", "legacy", "other"]
        sorted_categories = sorted(
            categories.keys(),
            key=lambda x: category_order.index(x) if x in category_order else len(category_order),
        )

        for category in sorted_categories:
            models = categories[category]
            lines.append(f"🔹 {category.replace('_', ' ').title()} Models:")

            # Sort by availability and name
            models.sort(key=lambda x: (not x[1]["available"], x[0]))

            for model_name, data in models:
                available = data["available"]
                latency = data.get("latency")
                last_checked = data.get("last_checked")

                status_icon = "✅" if available else "❌"
                latency_str = f", {latency:.2f}s" if latency else ""
                checked_str = f" (checked {last_checked})" if last_checked else ""

                lines.append(f"  {status_icon} {model_name}{latency_str}{checked_str}")

            lines.append("")

        # Summary stats
        total_models = len(status_data)
        available_models = sum(1 for data in status_data.values() if data["available"])
        tested_models = sum(1 for data in status_data.values() if data.get("latency") is not None)

        lines.extend(
            [
                "📈 Status Summary:",
                f"  • Total Models: {total_models}",
                f"  • Available: {available_models}",
                f"  • Tested Accessible: {tested_models}",
                f"  • Availability Rate: {available_models / total_models * 100:.1f}%"
                if total_models > 0
                else "  • Availability Rate: N/A",
                "",
            ]
        )

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

    def run_auto_config_agent(self, test_accessibility: bool = True) -> dict[str, Any]:
        """Run end-to-end automatic model configuration workflow.

        This agent performs all steps without manual intervention:
        1) fetch current API models,
        2) compare with configured models,
        3) resolve context limits via Hugging Face when possible,
        4) optionally test accessibility,
        5) persist status and search configuration artifacts.

        Args:
            test_accessibility: Whether to run live chat accessibility checks.

        Returns:
            Dictionary containing summary and generated artifact paths.
        """
        print("🤖 Running automatic model configuration agent...")

        api_models = self.get_api_models()
        raw_model_ids = [model.get("id") for model in api_models if isinstance(model, dict)]
        parsed_models = parse_api_model_ids(
            [model_id for model_id in raw_model_ids if isinstance(model_id, str)]
        )

        configured_models = self.get_configured_models()
        api_ids = {model.api_id for model in parsed_models}
        configured_ids = set(configured_models.keys())

        configured_not_available = sorted(configured_ids - api_ids)
        available_not_configured = sorted(api_ids - configured_ids)

        status_data: dict[str, Any] = {}
        search_models: list[dict[str, Any]] = []
        hf_found_count = 0
        accessible_count = 0

        for model in parsed_models:
            known = get_model_by_name(model.name)
            configured_limit = known.max_context_tokens if known else None
            hf_limit = _get_online_token_limit(model.name, "huggingface")
            if hf_limit is not None:
                hf_found_count += 1

            effective_limit = hf_limit or configured_limit or DEFAULT_TOKEN_LIMIT

            accessible = None
            latency = None
            if test_accessibility:
                print(f"  🧪 Testing {model.name}...")
                is_accessible, value_latency = self.test_model_accessibility(model.name)
                accessible = is_accessible
                latency = round(value_latency, 3) if is_accessible else None
                if is_accessible:
                    accessible_count += 1

            now_ts = time.time()
            now_dt = datetime.fromtimestamp(now_ts).isoformat(sep=" ", timespec="seconds")
            status_data[model.name] = {
                "name": model.name,
                "description": model.description,
                "tokens": effective_limit,
                "hf_tokens": hf_limit,
                "configured_tokens": configured_limit,
                "available": True,
                "accessible": accessible,
                "latency": latency,
                "category": self._categorize_model(model.name),
                "provider": "blablador",
                "api_id": model.api_id,
                "last_checked": now_ts,
                "last_checked_datetime": now_dt,
            }

            search_models.append(
                {
                    "name": model.name,
                    "api_id": model.api_id,
                    "context_tokens": effective_limit,
                    "hf_context_tokens": hf_limit,
                    "configured_context_tokens": configured_limit,
                    "accessible": accessible,
                    "latency": latency,
                    "aliases": [model.alias] if model.alias else [],
                    "description": model.description,
                }
            )

        # Save status in existing canonical status file.
        self.save_model_status(status_data)

        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        search_payload = {
            "generated_at": datetime.now().isoformat(sep=" ", timespec="seconds"),
            "api_models_count": len(parsed_models),
            "configured_models_count": len(configured_models),
            "configured_not_available": configured_not_available,
            "available_not_configured": available_not_configured,
            "hf_token_limits_found": hf_found_count,
            "accessibility_tested": test_accessibility,
            "accessible_models": accessible_count if test_accessibility else None,
            "models": sorted(search_models, key=lambda item: item["name"].lower()),
        }

        latest_path = reports_dir / "model_search_config_latest.yaml"
        timestamped_path = reports_dir / f"model_search_config_{timestamp}.yaml"
        with open(latest_path, "w", encoding="utf-8") as f:
            yaml.dump(search_payload, f, default_flow_style=False, sort_keys=False)
        with open(timestamped_path, "w", encoding="utf-8") as f:
            yaml.dump(search_payload, f, default_flow_style=False, sort_keys=False)

        summary_lines = [
            "🤖 Automatic Model Configuration Agent Report",
            "=" * 55,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "📈 Summary:",
            f"  • API Models: {len(parsed_models)}",
            f"  • Configured Models: {len(configured_models)}",
            f"  • Configured but Unavailable: {len(configured_not_available)}",
            f"  • Available but Unconfigured: {len(available_not_configured)}",
            f"  • HF Token Limits Found: {hf_found_count}/{len(parsed_models)}",
        ]
        if test_accessibility:
            summary_lines.append(f"  • Accessible Models: {accessible_count}/{len(parsed_models)}")

        if configured_not_available:
            summary_lines.append("")
            summary_lines.append("⚠️  Configured but currently unavailable:")
            for api_id in configured_not_available:
                summary_lines.append(f"  • {api_id}")

        if available_not_configured:
            summary_lines.append("")
            summary_lines.append("🔍 Available but not in static configuration:")
            for api_id in available_not_configured:
                summary_lines.append(f"  • {api_id}")

        summary_lines.append("")
        summary_lines.append(f"💾 Search config (latest): {latest_path}")
        summary_lines.append(f"💾 Search config (timestamped): {timestamped_path}")

        report_path = reports_dir / f"model_agent_report_{timestamp}.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(summary_lines))

        return {
            "report": "\n".join(summary_lines),
            "report_path": str(report_path),
            "latest_config_path": str(latest_path),
            "timestamped_config_path": str(timestamped_path),
        }


def monitor_models(
    test_accessibility: bool = False,
    save_report: bool = True,
    api_key: str | None = None,
    api_base: str | None = None,
    update_yaml: bool = False,
    enhanced_report: bool = False,
) -> str:
    """Convenience function to monitor model availability.

    Args:
        test_accessibility: Whether to test actual model accessibility.
        save_report: Whether to save the report to file.
        api_key: Optional API key override.
        api_base: Optional API base URL override.
        update_yaml: Whether to update the YAML status file with current availability.
        enhanced_report: Whether to generate enhanced report with YAML status data.

    Returns:
        The generated report content.
    """
    monitor = ModelAvailabilityMonitor(api_key=api_key, api_base=api_base)

    if update_yaml:
        print("🔄 Updating YAML model status...")
        monitor.check_all_models_automatically()

    if enhanced_report:
        report = monitor.generate_enhanced_report(include_yaml_status=True)
    else:
        analysis = monitor.analyze_availability(test_accessibility=test_accessibility)
        report = monitor.generate_report(analysis, test_accessibility=test_accessibility)

    if save_report:
        filepath = monitor.save_report(report)
        print(f"\n💾 Report saved to: {filepath}")

    return report
