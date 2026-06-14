"""Comprehensive tests for hellmholtz.monitoring module.

Targets >80% coverage of src/hellmholtz/monitoring.py by testing
ModelAvailabilityMonitor and monitor_models with full branch coverage.
"""

import os
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from hellmholtz.monitoring import ModelAvailabilityMonitor, monitor_models

# ---------------------------------------------------------------------------
# Helpers – lightweight stand-ins for real config objects
# ---------------------------------------------------------------------------


def _make_model(
    name: str,
    api_id: str | None = None,
    *,
    description: str = "desc",
    max_context_tokens: int = 8192,
    alias: str | None = None,
) -> MagicMock:
    """Return a MagicMock mimicking BlabladorModel."""
    m = MagicMock()
    m.name = name
    m.description = description
    m.max_context_tokens = max_context_tokens
    m.alias = alias
    m.api_id = api_id or name
    return m


FAKE_MODELS = [
    _make_model("GPT-OSS-120b", "1 - GPT-OSS-120b - GPT-OSS-120b model", description="big"),
    _make_model("ministral-3b", "2 - Ministral-3-14B - Ministral-3-14B model"),
    _make_model("llama-alias", "3 - llama-alias - legacy alias model"),
    _make_model("qwen-instruct", "4 - qwen-instruct - instruction tuned"),
    _make_model("phi-chat", "5 - phi-chat - conversational model"),
]


def _api_response(models: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    if models is None:
        models = [{"id": m.api_id} for m in FAKE_MODELS]
    return {"data": models}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure BLABLADOR env vars are unset before each test."""
    monkeypatch.delenv("BLABLADOR_API_KEY", raising=False)
    monkeypatch.delenv("BLABLADOR_API_BASE", raising=False)


def _monitor(api_key: str = "test_key", api_base: str | None = None) -> ModelAvailabilityMonitor:
    kwargs: dict[str, Any] = {"api_key": api_key}
    if api_base:
        kwargs["api_base"] = api_base
    return ModelAvailabilityMonitor(**kwargs)


# ===================================================================
# 1. __init__
# ===================================================================


class TestInit:
    def test_init_with_explicit_params(self) -> None:
        m = ModelAvailabilityMonitor(api_key="k", api_base="https://base.example.com/v1")
        assert m.api_key == "k"
        assert m.api_base == "https://base.example.com/v1"
        assert m.headers == {"Authorization": "Bearer k"}

    def test_init_with_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BLABLADOR_API_KEY", "env_key")
        monkeypatch.setenv("BLABLADOR_API_BASE", "https://env.example.com/v1")
        m = ModelAvailabilityMonitor()
        assert m.api_key == "env_key"
        assert m.api_base == "https://env.example.com/v1"

    def test_init_default_api_base(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BLABLADOR_API_KEY", "k")
        m = ModelAvailabilityMonitor()
        assert m.api_base == "https://api.blablador.example.com/v1"

    def test_init_missing_api_key_raises(self) -> None:
        with pytest.raises(ValueError, match="BLABLADOR_API_KEY not found"):
            ModelAvailabilityMonitor()

    def test_init_missing_api_key_with_none(self) -> None:
        with pytest.raises(ValueError, match="BLABLADOR_API_KEY not found"):
            ModelAvailabilityMonitor(api_key=None)

    def test_explicit_param_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BLABLADOR_API_KEY", "env_key")
        m = ModelAvailabilityMonitor(api_key="explicit")
        assert m.api_key == "explicit"

    def test_test_message_initialized(self) -> None:
        m = _monitor()
        assert m.test_message == [{"role": "user", "content": "Hello"}]


# ===================================================================
# 2. get_api_models
# ===================================================================


class TestGetApiModels:
    @patch("hellmholtz.monitoring.requests.get")
    def test_success(self, mock_get: MagicMock) -> None:
        mock_get.return_value = MagicMock(
            json=MagicMock(return_value=_api_response()),
            raise_for_status=MagicMock(),
        )
        models = _monitor().get_api_models()
        assert len(models) == 5
        mock_get.assert_called_once()

    @patch("hellmholtz.monitoring.requests.get")
    def test_success_empty_data(self, mock_get: MagicMock) -> None:
        mock_get.return_value = MagicMock(
            json=MagicMock(return_value={"data": []}),
            raise_for_status=MagicMock(),
        )
        assert _monitor().get_api_models() == []

    @patch("hellmholtz.monitoring.requests.get")
    def test_success_missing_data_key(self, mock_get: MagicMock) -> None:
        mock_get.return_value = MagicMock(
            json=MagicMock(return_value={}),
            raise_for_status=MagicMock(),
        )
        assert _monitor().get_api_models() == []

    @patch("hellmholtz.monitoring.requests.get")
    def test_request_exception(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = Exception("timeout")
        with pytest.raises(RuntimeError, match="Failed to fetch models from API"):
            _monitor().get_api_models()

    @patch("hellmholtz.monitoring.requests.get")
    def test_http_error(self, mock_get: MagicMock) -> None:
        import requests as _req

        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = _req.HTTPError("500")
        mock_resp.json.return_value = {}
        mock_get.return_value = mock_resp
        with pytest.raises(RuntimeError, match="Failed to fetch models from API"):
            _monitor().get_api_models()


# ===================================================================
# 3. get_configured_models
# ===================================================================


class TestGetConfiguredModels:
    @patch("hellmholtz.monitoring.KNOWN_MODELS", FAKE_MODELS)
    def test_returns_dict(self) -> None:
        result = _monitor().get_configured_models()
        assert isinstance(result, dict)
        assert len(result) == len(FAKE_MODELS)

    @patch("hellmholtz.monitoring.KNOWN_MODELS", FAKE_MODELS)
    def test_keys_are_api_ids(self) -> None:
        result = _monitor().get_configured_models()
        for m in FAKE_MODELS:
            assert m.api_id in result

    @patch("hellmholtz.monitoring.KNOWN_MODELS", [])
    def test_empty_known_models(self) -> None:
        assert _monitor().get_configured_models() == {}


# ===================================================================
# 4. test_model_accessibility
# ===================================================================


class TestModelAccessibility:
    @patch("hellmholtz.monitoring.time.time", side_effect=[1000.0, 1000.5])
    @patch("hellmholtz.monitoring.chat", return_value="Hi there")
    def test_success(self, mock_chat: MagicMock, _mock_time: MagicMock) -> None:
        accessible, latency = _monitor().test_model_accessibility("m")
        assert accessible is True
        assert latency == pytest.approx(0.5)
        mock_chat.assert_called_once_with(
            "blablador:m",
            [{"role": "user", "content": "Hello"}],
            max_tokens=5,
            timeout=10.0,
        )

    @patch("hellmholtz.monitoring.chat", return_value="")
    def test_empty_response(self, mock_chat: MagicMock) -> None:
        accessible, _ = _monitor().test_model_accessibility("m")
        assert accessible is False

    @patch("hellmholtz.monitoring.chat", return_value=None)
    def test_none_response(self, mock_chat: MagicMock) -> None:
        accessible, _ = _monitor().test_model_accessibility("m")
        assert accessible is False

    @patch("hellmholtz.monitoring.chat", side_effect=Exception("down"))
    def test_exception(self, mock_chat: MagicMock) -> None:
        accessible, latency = _monitor().test_model_accessibility("m")
        assert accessible is False
        assert latency == 0.0

    @patch("hellmholtz.monitoring.time.time", side_effect=[1000.0, 1001.2])
    @patch("hellmholtz.monitoring.chat", return_value="ok")
    def test_custom_timeout(self, mock_chat: MagicMock, _mock_time: MagicMock) -> None:
        _monitor().test_model_accessibility("m", timeout=30.0)
        mock_chat.assert_called_once_with(
            "blablador:m",
            [{"role": "user", "content": "Hello"}],
            max_tokens=5,
            timeout=30.0,
        )


# ===================================================================
# 5. analyze_availability
# ===================================================================


class TestAnalyzeAvailability:
    @patch("hellmholtz.monitoring.KNOWN_MODELS", FAKE_MODELS[:2])
    @patch("hellmholtz.monitoring.parse_api_model_ids")
    @patch("hellmholtz.monitoring.requests.get")
    def test_basic_analysis(
        self,
        mock_get: MagicMock,
        mock_parse: MagicMock,
    ) -> None:
        mock_get.return_value = MagicMock(
            json=MagicMock(return_value=_api_response([{"id": FAKE_MODELS[0].api_id}])),
            raise_for_status=MagicMock(),
        )
        parsed = [_make_model("GPT-OSS-120b", FAKE_MODELS[0].api_id)]
        mock_parse.return_value = parsed

        analysis = _monitor().analyze_availability(test_accessibility=False)

        assert "api_models_count" in analysis
        assert "configured_models_count" in analysis
        assert "configured_and_available" in analysis
        assert "configured_not_available" in analysis
        assert "available_not_configured" in analysis
        assert "accessibility_results" in analysis
        assert analysis["api_models_count"] == 1

    @patch("hellmholtz.monitoring.KNOWN_MODELS", FAKE_MODELS[:1])
    @patch("hellmholtz.monitoring.parse_api_model_ids")
    @patch("hellmholtz.monitoring.requests.get")
    def test_with_accessibility_testing(
        self,
        mock_get: MagicMock,
        mock_parse: MagicMock,
    ) -> None:
        mock_get.return_value = MagicMock(
            json=MagicMock(return_value=_api_response([{"id": FAKE_MODELS[0].api_id}])),
            raise_for_status=MagicMock(),
        )
        parsed = [_make_model("GPT-OSS-120b", FAKE_MODELS[0].api_id)]
        mock_parse.return_value = parsed

        with patch.object(
            ModelAvailabilityMonitor, "test_model_accessibility", return_value=(True, 0.1)
        ):
            analysis = _monitor().analyze_availability(test_accessibility=True)

        assert analysis["accessibility_results"]["GPT-OSS-120b"]["accessible"] is True

    @patch("hellmholtz.monitoring.KNOWN_MODELS", FAKE_MODELS[:1])
    @patch("hellmholtz.monitoring.parse_api_model_ids")
    @patch("hellmholtz.monitoring.requests.get")
    def test_configured_not_available(
        self,
        mock_get: MagicMock,
        mock_parse: MagicMock,
    ) -> None:
        mock_get.return_value = MagicMock(
            json=MagicMock(return_value=_api_response([])),
            raise_for_status=MagicMock(),
        )
        mock_parse.return_value = []

        analysis = _monitor().analyze_availability(test_accessibility=False)
        assert len(analysis["configured_not_available"]) == 1

    @patch("hellmholtz.monitoring.KNOWN_MODELS", [])
    @patch("hellmholtz.monitoring.parse_api_model_ids")
    @patch("hellmholtz.monitoring.requests.get")
    def test_available_not_configured(
        self,
        mock_get: MagicMock,
        mock_parse: MagicMock,
    ) -> None:
        extra = _make_model("extra-model", "99 - extra - extra model")
        mock_get.return_value = MagicMock(
            json=MagicMock(return_value=_api_response([{"id": extra.api_id}])),
            raise_for_status=MagicMock(),
        )
        mock_parse.return_value = [extra]

        analysis = _monitor().analyze_availability(test_accessibility=False)
        assert len(analysis["available_not_configured"]) == 1

    @patch("hellmholtz.monitoring.KNOWN_MODELS", FAKE_MODELS[:1])
    @patch("hellmholtz.monitoring.parse_api_model_ids")
    @patch("hellmholtz.monitoring.requests.get")
    def test_accessibility_not_accessible(
        self,
        mock_get: MagicMock,
        mock_parse: MagicMock,
    ) -> None:
        mock_get.return_value = MagicMock(
            json=MagicMock(return_value=_api_response([{"id": FAKE_MODELS[0].api_id}])),
            raise_for_status=MagicMock(),
        )
        parsed = [_make_model("GPT-OSS-120b", FAKE_MODELS[0].api_id)]
        mock_parse.return_value = parsed

        with patch.object(
            ModelAvailabilityMonitor, "test_model_accessibility", return_value=(False, 0.0)
        ):
            analysis = _monitor().analyze_availability(test_accessibility=True)

        assert analysis["accessibility_results"]["GPT-OSS-120b"]["accessible"] is False

    @patch("hellmholtz.monitoring.KNOWN_MODELS", FAKE_MODELS[:1])
    @patch("hellmholtz.monitoring.parse_api_model_ids")
    @patch("hellmholtz.monitoring.requests.get")
    def test_raw_model_ids_filter_non_dict(
        self,
        mock_get: MagicMock,
        mock_parse: MagicMock,
    ) -> None:
        raw = [{"id": FAKE_MODELS[0].api_id}, "not_a_dict", {"no_id_key": True}]
        mock_get.return_value = MagicMock(
            json=MagicMock(return_value={"data": raw}),
            raise_for_status=MagicMock(),
        )
        mock_parse.return_value = [_make_model("GPT-OSS-120b", FAKE_MODELS[0].api_id)]

        analysis = _monitor().analyze_availability(test_accessibility=False)
        assert isinstance(analysis, dict)


# ===================================================================
# 6. generate_report
# ===================================================================


class TestGenerateReport:
    def _analysis(self, **overrides: Any) -> dict[str, Any]:
        base: dict[str, Any] = {
            "api_models_count": 10,
            "configured_models_count": 5,
            "configured_and_available": [],
            "configured_not_available": [],
            "available_not_configured": [],
            "accessibility_results": {},
            "timestamp": time.time(),
        }
        base.update(overrides)
        return base

    def test_report_with_all_sections(self) -> None:
        m = _make_model("gpt-4o", "1 - gpt-4o - OpenAI model")
        analysis = self._analysis(
            configured_and_available=[("1 - gpt-4o - OpenAI model", m)],
            configured_not_available=[("2 - old - old model", m)],
            available_not_configured=[("3 - new", {"id": "3 - new", "object": "model"})],
        )
        report = _monitor().generate_report(analysis, test_accessibility=False)
        assert "Blablador Model Availability Report" in report
        assert "Available & Configured" in report
        assert "Configured but Not Available" in report
        assert "Available but Not Configured" in report

    def test_report_empty_analysis(self) -> None:
        report = _monitor().generate_report(self._analysis())
        assert "Configuration is up-to-date" in report

    def test_report_with_accessibility(self) -> None:
        m = _make_model("gpt-4o", "1 - gpt-4o")
        analysis = self._analysis(
            configured_and_available=[("1 - gpt-4o", m)],
            accessibility_results={"gpt-4o": {"accessible": True, "latency": 0.5, "api_id": "x"}},
        )
        report = _monitor().generate_report(analysis, test_accessibility=True)
        assert "Accessible" in report

    def test_report_with_inaccessible_model(self) -> None:
        m = _make_model("gpt-4o", "1 - gpt-4o")
        analysis = self._analysis(
            configured_and_available=[("1 - gpt-4o", m)],
            accessibility_results={"gpt-4o": {"accessible": False, "latency": 0.0, "api_id": "x"}},
        )
        report = _monitor().generate_report(analysis, test_accessibility=True)
        assert "Not accessible" in report


# ===================================================================
# 7. _generate_report_header / _summary / _sections / _recommendations
# ===================================================================


class TestReportHelpers:
    def _analysis(self, **overrides: Any) -> dict[str, Any]:
        base: dict[str, Any] = {
            "api_models_count": 10,
            "configured_models_count": 5,
            "configured_and_available": [],
            "configured_not_available": [],
            "available_not_configured": [],
            "accessibility_results": {},
            "timestamp": time.time(),
        }
        base.update(overrides)
        return base

    def test_report_header(self) -> None:
        lines = _monitor()._generate_report_header(self._analysis())
        assert any("Blablador Model Availability Report" in l for l in lines)
        assert any("Generated:" in l for l in lines)

    def test_report_summary(self) -> None:
        lines = _monitor()._generate_report_summary(self._analysis())
        assert any("API Models: 10" in l for l in lines)
        assert any("Configured Models: 5" in l for l in lines)

    def test_report_sections_empty(self) -> None:
        lines = _monitor()._generate_report_sections(self._analysis(), False)
        assert lines == []

    def test_report_sections_all_populated(self) -> None:
        m = _make_model("x", "x-id")
        a = self._analysis(
            configured_and_available=[("x-id", m)],
            configured_not_available=[("y-id", m)],
            available_not_configured=[("z-id", {"id": "z-id", "object": "model"})],
        )
        lines = _monitor()._generate_report_sections(a, False)
        text = "\n".join(lines)
        assert "Available & Configured" in text
        assert "Configured but Not Available" in text
        assert "Available but Not Configured" in text

    def test_recommendations_unavailable(self) -> None:
        m = _make_model("x", "x-id")
        a = self._analysis(configured_not_available=[("x-id", m)])
        lines = _monitor()._generate_report_recommendations(a)
        text = "\n".join(lines)
        assert "Review configured models" in text

    def test_recommendations_unconfigured(self) -> None:
        a = self._analysis(available_not_configured=[("z", {"id": "z", "object": "m"})])
        lines = _monitor()._generate_report_recommendations(a)
        text = "\n".join(lines)
        assert "Consider adding" in text

    def test_recommendations_both(self) -> None:
        m = _make_model("x", "x-id")
        a = self._analysis(
            configured_not_available=[("x-id", m)],
            available_not_configured=[("z", {"id": "z", "object": "m"})],
        )
        lines = _monitor()._generate_report_recommendations(a)
        text = "\n".join(lines)
        assert "Review configured models" in text
        assert "Consider adding" in text

    def test_recommendations_up_to_date(self) -> None:
        lines = _monitor()._generate_report_recommendations(self._analysis())
        text = "\n".join(lines)
        assert "up-to-date" in text


# ===================================================================
# 8. load_model_status
# ===================================================================


class TestLoadModelStatus:
    def test_missing_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        assert _monitor().load_model_status() == {}

    def test_valid_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "models_status.yaml").write_text("models:\n  m1:\n    available: true\n")
        result = _monitor().load_model_status()
        assert "m1" in result
        assert result["m1"]["available"] is True

    def test_valid_file_flat(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "models_status.yaml").write_text("m1:\n  available: true\n")
        result = _monitor().load_model_status()
        assert "m1" in result

    def test_corrupt_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "models_status.yaml").write_text("{{invalid yaml::")
        result = _monitor().load_model_status()
        assert result == {}

    def test_non_dict_content(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "models_status.yaml").write_text("- item1\n- item2\n")
        result = _monitor().load_model_status()
        assert result == {}

    def test_empty_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "models_status.yaml").write_text("")
        result = _monitor().load_model_status()
        assert result == {}


# ===================================================================
# 9. save_model_status
# ===================================================================


class TestSaveModelStatus:
    def test_new_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        data = {"m1": {"available": True, "latency": 0.1}}
        _monitor().save_model_status(data)
        assert (tmp_path / "models_status.yaml").exists()
        import yaml

        loaded = yaml.safe_load((tmp_path / "models_status.yaml").read_text())
        assert "models" in loaded
        assert loaded["models"]["m1"]["available"] is True
        assert loaded["# Total models"] == 1
        assert loaded["# Working"] == 1
        assert loaded["# Broken"] == 0

    def test_existing_file_preserves_models_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "models_status.yaml").write_text(
            "models:\n  old_model:\n    available: false\n"
        )
        _monitor().save_model_status({"new_model": {"available": True}})
        import yaml

        loaded = yaml.safe_load((tmp_path / "models_status.yaml").read_text())
        assert "new_model" in loaded["models"]
        assert "old_model" not in loaded["models"]

    def test_all_unavailable(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        data = {"m1": {"available": False}, "m2": {"available": False}}
        _monitor().save_model_status(data)
        import yaml

        loaded = yaml.safe_load((tmp_path / "models_status.yaml").read_text())
        assert loaded["# Working"] == 0
        assert loaded["# Broken"] == 2

    def test_corrupt_existing_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "models_status.yaml").write_text("{{{{invalid yaml::")
        _monitor().save_model_status({"m1": {"available": True}})
        import yaml

        loaded = yaml.safe_load((tmp_path / "models_status.yaml").read_text())
        assert "m1" in loaded["models"]

    def test_write_failure(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        with patch("builtins.open", side_effect=PermissionError("denied")):
            # Should not raise — error is logged
            _monitor().save_model_status({"m1": {"available": True}})

    def test_metadata_fields(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        _monitor().save_model_status({"a": {"available": True}})
        import yaml

        loaded = yaml.safe_load((tmp_path / "models_status.yaml").read_text())
        assert "# Last updated" in loaded
        assert "# This file is automatically updated by the model availability checker" in loaded


# ===================================================================
# 10. check_all_models_automatically
# ===================================================================


class TestCheckAllModelsAutomatically:
    @patch("hellmholtz.monitoring.KNOWN_MODELS", FAKE_MODELS[:1])
    @patch("hellmholtz.monitoring.requests.get")
    def test_full_workflow(
        self,
        mock_get: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        mock_get.return_value = MagicMock(
            json=MagicMock(return_value=_api_response([{"id": FAKE_MODELS[0].api_id}])),
            raise_for_status=MagicMock(),
        )
        with patch.object(
            ModelAvailabilityMonitor,
            "test_model_accessibility",
            return_value=(True, 0.123),
        ):
            result = _monitor().check_all_models_automatically()

        assert FAKE_MODELS[0].name in result
        entry = result[FAKE_MODELS[0].name]
        assert entry["available"] is True
        assert entry["latency"] == pytest.approx(0.123)
        assert entry["last_checked"] is not None
        assert entry["last_checked_datetime"] is not None

    @patch("hellmholtz.monitoring.KNOWN_MODELS", FAKE_MODELS[:1])
    @patch("hellmholtz.monitoring.requests.get")
    def test_api_fetch_failure(
        self,
        mock_get: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        mock_get.side_effect = Exception("down")
        result = _monitor().check_all_models_automatically()
        # Should return empty dict since no previous status
        assert isinstance(result, dict)

    @patch("hellmholtz.monitoring.KNOWN_MODELS", FAKE_MODELS[:1])
    @patch("hellmholtz.monitoring.requests.get")
    def test_model_not_in_api(
        self,
        mock_get: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        mock_get.return_value = MagicMock(
            json=MagicMock(return_value=_api_response([])),
            raise_for_status=MagicMock(),
        )
        result = _monitor().check_all_models_automatically()
        entry = result[FAKE_MODELS[0].name]
        assert entry["available"] is False
        assert entry["latency"] is None

    @patch("hellmholtz.monitoring.KNOWN_MODELS", FAKE_MODELS[:1])
    @patch("hellmholtz.monitoring.requests.get")
    def test_model_accessible_but_fails_test(
        self,
        mock_get: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        mock_get.return_value = MagicMock(
            json=MagicMock(return_value=_api_response([{"id": FAKE_MODELS[0].api_id}])),
            raise_for_status=MagicMock(),
        )
        with patch.object(
            ModelAvailabilityMonitor,
            "test_model_accessibility",
            return_value=(False, 0.0),
        ):
            result = _monitor().check_all_models_automatically()

        entry = result[FAKE_MODELS[0].name]
        assert entry["latency"] is None

    @patch("hellmholtz.monitoring.KNOWN_MODELS", FAKE_MODELS[:1])
    @patch("hellmholtz.monitoring.requests.get")
    def test_loads_existing_status(
        self,
        mock_get: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "models_status.yaml").write_text(
            "models:\n  pre_existing:\n    available: true\n"
        )
        mock_get.return_value = MagicMock(
            json=MagicMock(return_value=_api_response([{"id": FAKE_MODELS[0].api_id}])),
            raise_for_status=MagicMock(),
        )
        with patch.object(
            ModelAvailabilityMonitor,
            "test_model_accessibility",
            return_value=(True, 0.05),
        ):
            result = _monitor().check_all_models_automatically()

        assert "pre_existing" in result

    @patch("hellmholtz.monitoring.KNOWN_MODELS", FAKE_MODELS[:1])
    @patch("hellmholtz.monitoring.requests.get")
    def test_model_key_already_in_status(
        self,
        mock_get: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        import yaml

        existing = {
            "models": {
                FAKE_MODELS[0].name: {
                    "name": FAKE_MODELS[0].name,
                    "available": False,
                    "latency": None,
                    "category": "other",
                    "last_checked": None,
                }
            }
        }
        (tmp_path / "models_status.yaml").write_text(yaml.dump(existing))
        mock_get.return_value = MagicMock(
            json=MagicMock(return_value=_api_response([{"id": FAKE_MODELS[0].api_id}])),
            raise_for_status=MagicMock(),
        )
        with patch.object(
            ModelAvailabilityMonitor,
            "test_model_accessibility",
            return_value=(True, 0.05),
        ):
            result = _monitor().check_all_models_automatically()

        assert result[FAKE_MODELS[0].name]["available"] is True


# ===================================================================
# 11. _categorize_model
# ===================================================================


class TestCategorizeModel:
    def test_alias(self) -> None:
        assert _monitor()._categorize_model("my-alias-model") == "alias"

    def test_legacy(self) -> None:
        assert _monitor()._categorize_model("legacy-v1") == "legacy"

    def test_old(self) -> None:
        assert _monitor()._categorize_model("old-model-3b") == "legacy"

    def test_base_model_3b(self) -> None:
        assert _monitor()._categorize_model("phi-3b") == "base_model"

    def test_base_model_7b(self) -> None:
        assert _monitor()._categorize_model("llama-7b") == "base_model"

    def test_base_model_14b(self) -> None:
        assert _monitor()._categorize_model("model-14b") == "base_model"

    def test_base_model_32b(self) -> None:
        assert _monitor()._categorize_model("model-32b") == "base_model"

    def test_base_model_70b(self) -> None:
        assert _monitor()._categorize_model("model-70b") == "base_model"

    def test_base_model_120b(self) -> None:
        assert _monitor()._categorize_model("model-120b") == "base_model"

    def test_base_model_405b(self) -> None:
        assert _monitor()._categorize_model("model-405b") == "base_model"

    def test_instruction_tuned_instruct(self) -> None:
        assert _monitor()._categorize_model("gpt-instruct") == "instruction_tuned"

    def test_instruction_tuned_chat(self) -> None:
        assert _monitor()._categorize_model("gpt-chat") == "instruction_tuned"

    def test_instruction_tuned_conversational(self) -> None:
        assert _monitor()._categorize_model("gpt-conversational") == "instruction_tuned"

    def test_other(self) -> None:
        assert _monitor()._categorize_model("mysterious-model") == "other"

    def test_case_insensitive(self) -> None:
        assert _monitor()._categorize_model("ALIAS-MODEL") == "alias"

    def test_base_model_priority_over_instruct(self) -> None:
        # "7b" appears before "instruct" check but base_model check comes first
        assert _monitor()._categorize_model("model-7b-instruct") == "base_model"

    def test_alias_priority_over_old(self) -> None:
        # "alias" check is first
        assert _monitor()._categorize_model("old-alias-model") == "alias"


# ===================================================================
# 12. generate_enhanced_report
# ===================================================================


class TestGenerateEnhancedReport:
    @patch("hellmholtz.monitoring.KNOWN_MODELS", FAKE_MODELS[:1])
    @patch("hellmholtz.monitoring.parse_api_model_ids")
    @patch("hellmholtz.monitoring.requests.get")
    def test_with_yaml_data(
        self,
        mock_get: MagicMock,
        mock_parse: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "models_status.yaml").write_text(
            "models:\n  m1:\n    available: true\n    latency: 0.5\n    category: base_model\n"
        )
        mock_get.return_value = MagicMock(
            json=MagicMock(return_value=_api_response([])),
            raise_for_status=MagicMock(),
        )
        mock_parse.return_value = []

        report = _monitor().generate_enhanced_report(include_yaml_status=True)
        assert "Enhanced Blablador Model Status Report" in report
        assert "Model Status from YAML" in report

    @patch("hellmholtz.monitoring.KNOWN_MODELS", FAKE_MODELS[:1])
    @patch("hellmholtz.monitoring.parse_api_model_ids")
    @patch("hellmholtz.monitoring.requests.get")
    def test_without_yaml_data(
        self,
        mock_get: MagicMock,
        mock_parse: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        mock_get.return_value = MagicMock(
            json=MagicMock(return_value=_api_response([])),
            raise_for_status=MagicMock(),
        )
        mock_parse.return_value = []

        report = _monitor().generate_enhanced_report(include_yaml_status=True)
        assert "No YAML status data found" in report

    @patch("hellmholtz.monitoring.KNOWN_MODELS", FAKE_MODELS[:1])
    @patch("hellmholtz.monitoring.parse_api_model_ids")
    @patch("hellmholtz.monitoring.requests.get")
    def test_without_yaml_status(
        self,
        mock_get: MagicMock,
        mock_parse: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        mock_get.return_value = MagicMock(
            json=MagicMock(return_value=_api_response([])),
            raise_for_status=MagicMock(),
        )
        mock_parse.return_value = []

        report = _monitor().generate_enhanced_report(include_yaml_status=False)
        assert "Enhanced Blablador Model Status Report" in report
        assert "Model Status from YAML" not in report

    @patch("hellmholtz.monitoring.KNOWN_MODELS", FAKE_MODELS[:1])
    @patch("hellmholtz.monitoring.requests.get")
    def test_analysis_error(
        self,
        mock_get: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        mock_get.side_effect = Exception("boom")
        report = _monitor().generate_enhanced_report(include_yaml_status=False)
        assert "Error generating analysis" in report


# ===================================================================
# 13. _generate_yaml_status_section
# ===================================================================


class TestGenerateYamlStatusSection:
    def test_empty_data(self) -> None:
        lines = _monitor()._generate_yaml_status_section({})
        assert "Model Status from YAML" in lines[0]

    def test_single_category(self) -> None:
        data = {
            "m1": {
                "available": True,
                "latency": 0.5,
                "category": "base_model",
                "last_checked": 1000.0,
            }
        }
        lines = _monitor()._generate_yaml_status_section(data)
        text = "\n".join(lines)
        assert "Base Model" in text
        assert "m1" in text

    def test_multiple_categories_sorted(self) -> None:
        data = {
            "m_alias": {"available": True, "latency": None, "category": "alias"},
            "m_base": {"available": False, "latency": None, "category": "base_model"},
            "m_other": {"available": True, "latency": 0.1, "category": "other"},
        }
        lines = _monitor()._generate_yaml_status_section(data)
        text = "\n".join(lines)
        # base_model should appear before alias, alias before other
        base_pos = text.index("Base Model")
        alias_pos = text.index("Alias")
        other_pos = text.index("Other")
        assert base_pos < alias_pos < other_pos

    def test_summary_stats(self) -> None:
        data = {
            "m1": {"available": True, "latency": 0.1, "category": "other"},
            "m2": {"available": False, "latency": None, "category": "other"},
        }
        lines = _monitor()._generate_yaml_status_section(data)
        text = "\n".join(lines)
        assert "Total Models: 2" in text
        assert "Available: 1" in text
        assert "Availability Rate: 50.0%" in text

    def test_zero_models_availability_rate(self) -> None:
        lines = _monitor()._generate_yaml_status_section({})
        text = "\n".join(lines)
        assert "Availability Rate: N/A" in text

    def test_unknown_category(self) -> None:
        data = {"m1": {"available": True, "category": "custom_cat"}}
        lines = _monitor()._generate_yaml_status_section(data)
        text = "\n".join(lines)
        assert "Custom Cat" in text

    def test_sorting_within_category(self) -> None:
        data = {
            "m_broken": {"available": False, "category": "base_model"},
            "m_ok": {"available": True, "category": "base_model"},
        }
        lines = _monitor()._generate_yaml_status_section(data)
        text = "\n".join(lines)
        ok_pos = text.index("m_ok")
        broken_pos = text.index("m_broken")
        assert ok_pos < broken_pos


# ===================================================================
# 14. save_report
# ===================================================================


class TestSaveReport:
    def test_with_explicit_filename(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        filepath = _monitor().save_report("report content", "my_report.txt")
        assert filepath.endswith("my_report.txt")
        assert Path(filepath).read_text() == "report content"
        assert (tmp_path / "reports" / "my_report.txt").exists()

    def test_with_none_filename(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        filepath = _monitor().save_report("content")
        assert "model_availability_report_" in filepath
        assert Path(filepath).exists()

    def test_creates_reports_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        assert not (tmp_path / "reports").exists()
        _monitor().save_report("x")
        assert (tmp_path / "reports").is_dir()


# ===================================================================
# 15. run_auto_config_agent
# ===================================================================


class TestRunAutoConfigAgent:
    @patch("hellmholtz.monitoring.KNOWN_MODELS", FAKE_MODELS[:2])
    @patch("hellmholtz.monitoring.get_model_by_name", return_value=None)
    @patch("hellmholtz.monitoring._get_online_token_limit", return_value=None)
    @patch("hellmholtz.monitoring.parse_api_model_ids")
    @patch("hellmholtz.monitoring.requests.get")
    def test_full_workflow(
        self,
        mock_get: MagicMock,
        mock_parse: MagicMock,
        mock_hf: MagicMock,
        mock_known: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        mock_get.return_value = MagicMock(
            json=MagicMock(
                return_value=_api_response(
                    [{"id": FAKE_MODELS[0].api_id}, {"id": FAKE_MODELS[1].api_id}]
                )
            ),
            raise_for_status=MagicMock(),
        )
        mock_parse.return_value = FAKE_MODELS[:2]

        with patch.object(
            ModelAvailabilityMonitor,
            "test_model_accessibility",
            return_value=(True, 0.2),
        ):
            result = _monitor().run_auto_config_agent(test_accessibility=True)

        assert "report" in result
        assert "report_path" in result
        assert "latest_config_path" in result
        assert "timestamped_config_path" in result
        assert Path(result["report_path"]).exists()

    @patch("hellmholtz.monitoring.KNOWN_MODELS", FAKE_MODELS[:1])
    @patch("hellmholtz.monitoring.get_model_by_name", return_value=None)
    @patch("hellmholtz.monitoring._get_online_token_limit", return_value=None)
    @patch("hellmholtz.monitoring.parse_api_model_ids")
    @patch("hellmholtz.monitoring.requests.get")
    def test_without_accessibility(
        self,
        mock_get: MagicMock,
        mock_parse: MagicMock,
        mock_hf: MagicMock,
        mock_known: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        mock_get.return_value = MagicMock(
            json=MagicMock(return_value=_api_response([{"id": FAKE_MODELS[0].api_id}])),
            raise_for_status=MagicMock(),
        )
        mock_parse.return_value = FAKE_MODELS[:1]

        result = _monitor().run_auto_config_agent(test_accessibility=False)
        assert "Accessible Models" not in result["report"]

    @patch("hellmholtz.monitoring.KNOWN_MODELS", FAKE_MODELS[:1])
    @patch("hellmholtz.monitoring.get_model_by_name", return_value=None)
    @patch("hellmholtz.monitoring._get_online_token_limit")
    @patch("hellmholtz.monitoring.parse_api_model_ids")
    @patch("hellmholtz.monitoring.requests.get")
    def test_with_hf_token_limit(
        self,
        mock_get: MagicMock,
        mock_parse: MagicMock,
        mock_hf: MagicMock,
        mock_known: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        model = _make_model("GPT-OSS-120b", FAKE_MODELS[0].api_id)
        mock_get.return_value = MagicMock(
            json=MagicMock(return_value=_api_response([{"id": model.api_id}])),
            raise_for_status=MagicMock(),
        )
        mock_parse.return_value = [model]
        mock_hf.return_value = 128000

        result = _monitor().run_auto_config_agent(test_accessibility=False)
        assert "HF Token Limits Found: 1/1" in result["report"]

    @patch("hellmholtz.monitoring.KNOWN_MODELS", FAKE_MODELS[:1])
    @patch("hellmholtz.monitoring.get_model_by_name", return_value=None)
    @patch("hellmholtz.monitoring._get_online_token_limit", return_value=None)
    @patch("hellmholtz.monitoring.parse_api_model_ids")
    @patch("hellmholtz.monitoring.requests.get")
    def test_configured_not_available_in_report(
        self,
        mock_get: MagicMock,
        mock_parse: MagicMock,
        mock_hf: MagicMock,
        mock_known: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        other = _make_model("other-model", "99 - other - other")
        mock_get.return_value = MagicMock(
            json=MagicMock(return_value=_api_response([{"id": other.api_id}])),
            raise_for_status=MagicMock(),
        )
        mock_parse.return_value = [other]

        result = _monitor().run_auto_config_agent(test_accessibility=False)
        assert "Configured but currently unavailable" in result["report"]

    @patch("hellmholtz.monitoring.KNOWN_MODELS", FAKE_MODELS[:1])
    @patch("hellmholtz.monitoring.get_model_by_name", return_value=None)
    @patch("hellmholtz.monitoring._get_online_token_limit", return_value=None)
    @patch("hellmholtz.monitoring.parse_api_model_ids")
    @patch("hellmholtz.monitoring.requests.get")
    def test_available_not_configured_in_report(
        self,
        mock_get: MagicMock,
        mock_parse: MagicMock,
        mock_hf: MagicMock,
        mock_known: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        extra = _make_model("extra", "999 - extra - extra")
        mock_get.return_value = MagicMock(
            json=MagicMock(return_value=_api_response([{"id": extra.api_id}])),
            raise_for_status=MagicMock(),
        )
        mock_parse.return_value = [extra]

        result = _monitor().run_auto_config_agent(test_accessibility=False)
        assert "Available but not in static configuration" in result["report"]


# ===================================================================
# 16. monitor_models (convenience function)
# ===================================================================


class TestMonitorModels:
    @patch("hellmholtz.monitoring.KNOWN_MODELS", FAKE_MODELS[:1])
    @patch("hellmholtz.monitoring.parse_api_model_ids")
    @patch("hellmholtz.monitoring.requests.get")
    def test_basic_call(
        self,
        mock_get: MagicMock,
        mock_parse: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        mock_get.return_value = MagicMock(
            json=MagicMock(return_value=_api_response([])),
            raise_for_status=MagicMock(),
        )
        mock_parse.return_value = []

        report = monitor_models(api_key="k", save_report=False)
        assert isinstance(report, str)
        assert len(report) > 0

    @patch("hellmholtz.monitoring.KNOWN_MODELS", FAKE_MODELS[:1])
    @patch("hellmholtz.monitoring.parse_api_model_ids")
    @patch("hellmholtz.monitoring.requests.get")
    def test_with_save_report(
        self,
        mock_get: MagicMock,
        mock_parse: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        mock_get.return_value = MagicMock(
            json=MagicMock(return_value=_api_response([])),
            raise_for_status=MagicMock(),
        )
        mock_parse.return_value = []

        report = monitor_models(api_key="k", save_report=True)
        assert isinstance(report, str)
        assert (tmp_path / "reports").is_dir()

    @patch("hellmholtz.monitoring.KNOWN_MODELS", FAKE_MODELS[:1])
    @patch("hellmholtz.monitoring.parse_api_model_ids")
    @patch("hellmholtz.monitoring.requests.get")
    def test_with_test_accessibility(
        self,
        mock_get: MagicMock,
        mock_parse: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        mock_get.return_value = MagicMock(
            json=MagicMock(return_value=_api_response([])),
            raise_for_status=MagicMock(),
        )
        mock_parse.return_value = []

        with patch.object(
            ModelAvailabilityMonitor,
            "test_model_accessibility",
            return_value=(True, 0.1),
        ):
            report = monitor_models(
                api_key="k", save_report=False, test_accessibility=True
            )
        assert isinstance(report, str)

    @patch("hellmholtz.monitoring.KNOWN_MODELS", FAKE_MODELS[:1])
    @patch("hellmholtz.monitoring.parse_api_model_ids")
    @patch("hellmholtz.monitoring.requests.get")
    def test_with_update_yaml(
        self,
        mock_get: MagicMock,
        mock_parse: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        mock_get.return_value = MagicMock(
            json=MagicMock(return_value=_api_response([{"id": FAKE_MODELS[0].api_id}])),
            raise_for_status=MagicMock(),
        )
        mock_parse.return_value = FAKE_MODELS[:1]

        with patch.object(
            ModelAvailabilityMonitor,
            "test_model_accessibility",
            return_value=(True, 0.1),
        ):
            report = monitor_models(
                api_key="k", save_report=False, update_yaml=True
            )
        assert isinstance(report, str)

    @patch("hellmholtz.monitoring.KNOWN_MODELS", FAKE_MODELS[:1])
    @patch("hellmholtz.monitoring.parse_api_model_ids")
    @patch("hellmholtz.monitoring.requests.get")
    def test_with_enhanced_report(
        self,
        mock_get: MagicMock,
        mock_parse: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        mock_get.return_value = MagicMock(
            json=MagicMock(return_value=_api_response([])),
            raise_for_status=MagicMock(),
        )
        mock_parse.return_value = []

        report = monitor_models(
            api_key="k", save_report=False, enhanced_report=True
        )
        assert "Enhanced" in report

    @patch("hellmholtz.monitoring.KNOWN_MODELS", FAKE_MODELS[:1])
    @patch("hellmholtz.monitoring.parse_api_model_ids")
    @patch("hellmholtz.monitoring.requests.get")
    def test_with_enhanced_and_update_yaml(
        self,
        mock_get: MagicMock,
        mock_parse: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        mock_get.return_value = MagicMock(
            json=MagicMock(return_value=_api_response([{"id": FAKE_MODELS[0].api_id}])),
            raise_for_status=MagicMock(),
        )
        mock_parse.return_value = FAKE_MODELS[:1]

        with patch.object(
            ModelAvailabilityMonitor,
            "test_model_accessibility",
            return_value=(True, 0.1),
        ):
            report = monitor_models(
                api_key="k",
                save_report=False,
                update_yaml=True,
                enhanced_report=True,
            )
        assert "Enhanced" in report

    @patch("hellmholtz.monitoring.KNOWN_MODELS", FAKE_MODELS[:1])
    @patch("hellmholtz.monitoring.parse_api_model_ids")
    @patch("hellmholtz.monitoring.requests.get")
    def test_default_api_base_used(
        self,
        mock_get: MagicMock,
        mock_parse: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        mock_get.return_value = MagicMock(
            json=MagicMock(return_value=_api_response([])),
            raise_for_status=MagicMock(),
        )
        mock_parse.return_value = []

        monitor_models(api_key="k", save_report=False, api_base="https://custom.example.com/v1")
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert "custom.example.com" in call_args[0][0]


# ===================================================================
# 17. Edge cases and integration-style tests
# ===================================================================


class TestEdgeCases:
    def test_monitor_test_message_constant(self) -> None:
        m = _monitor()
        assert m.test_message == [{"role": "user", "content": "Hello"}]
        # Should be a fixed test message
        m2 = _monitor(api_key="other")
        assert m2.test_message == m.test_message

    @patch("hellmholtz.monitoring.KNOWN_MODELS", FAKE_MODELS)
    def test_configured_models_preserves_all_entries(self) -> None:
        result = _monitor().get_configured_models()
        assert len(result) == len(FAKE_MODELS)
        for model in FAKE_MODELS:
            assert model.api_id in result
            assert result[model.api_id] is model

    @patch("hellmholtz.monitoring.KNOWN_MODELS", FAKE_MODELS[:1])
    @patch("hellmholtz.monitoring.parse_api_model_ids")
    @patch("hellmholtz.monitoring.requests.get")
    def test_analyze_availability_preserves_api_model_data(
        self,
        mock_get: MagicMock,
        mock_parse: MagicMock,
    ) -> None:
        raw_model = {"id": FAKE_MODELS[0].api_id, "object": "model", "owned_by": "org"}
        mock_get.return_value = MagicMock(
            json=MagicMock(return_value={"data": [raw_model]}),
            raise_for_status=MagicMock(),
        )
        parsed = [_make_model("GPT-OSS-120b", FAKE_MODELS[0].api_id)]
        mock_parse.return_value = parsed

        analysis = _monitor().analyze_availability(test_accessibility=False)
        assert analysis["available_not_configured"] == [] or isinstance(
            analysis["available_not_configured"][0][1], dict
        )

    @patch("hellmholtz.monitoring.KNOWN_MODELS", FAKE_MODELS[:1])
    @patch("hellmholtz.monitoring.requests.get")
    def test_check_all_models_sets_last_checked_datetime(
        self,
        mock_get: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        mock_get.return_value = MagicMock(
            json=MagicMock(return_value=_api_response([{"id": FAKE_MODELS[0].api_id}])),
            raise_for_status=MagicMock(),
        )
        with patch.object(
            ModelAvailabilityMonitor,
            "test_model_accessibility",
            return_value=(True, 0.1),
        ):
            result = _monitor().check_all_models_automatically()

        dt_str = result[FAKE_MODELS[0].name]["last_checked_datetime"]
        assert "T" in dt_str or " " in dt_str  # ISO-ish format
