"""Unit tests for hellmholtz.diagnostics (hellm doctor check functions)."""

from __future__ import annotations

import json
import urllib.error

import pytest

from hellmholtz import diagnostics as dg


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("http://x", code, "err", {}, None)  # type: ignore[arg-type]


class TestHelpers:
    def test_api_paths_appends_v1_when_missing(self) -> None:
        models, chat = dg.api_paths("https://api.example.de")
        assert models == "https://api.example.de/v1/models"
        assert chat == "https://api.example.de/v1/chat/completions"

    def test_api_paths_keeps_existing_v1(self) -> None:
        models, chat = dg.api_paths("https://lite.example.de:4000/v1/")
        assert models == "https://lite.example.de:4000/v1/models"
        assert chat == "https://lite.example.de:4000/v1/chat/completions"

    def test_probe_model_name_strips_provider_prefix(self) -> None:
        assert dg.probe_model_name("blablador:alias-fast") == "alias-fast"
        assert dg.probe_model_name("alias-fast") == "alias-fast"

    def test_mask_never_leaks_whole_secret(self) -> None:
        masked = dg._mask("glpa-super-secret-value-rwu")
        assert "secret" not in masked
        assert masked.startswith("glpa")

    def test_check_result_status_matrix(self) -> None:
        assert dg.CheckResult("a", True, "x").status == "OK"
        assert dg.CheckResult("a", False, "x").status == "FAIL"
        assert dg.CheckResult("a", False, "x", warn=True).status == "WARN"


class TestStaticChecks:
    def test_python_ok(self) -> None:
        assert dg.check_python((3, 8)).ok

    def test_python_too_old(self) -> None:
        assert not dg.check_python((99, 0)).ok

    def test_mcp_extra_installed_or_warn(self) -> None:
        result = dg.check_mcp_extra()
        assert result.ok or result.warn  # never a hard failure

    def test_credentials_set_and_masked(self) -> None:
        result = dg.check_credentials("glpa-abcdefgh-ijkl")
        assert result.ok
        assert "abcdefgh" not in result.detail

    def test_credentials_missing(self) -> None:
        assert not dg.check_credentials(None).ok


class TestCheckModelAvailable:
    def _body(self, ids: list[str]) -> str:
        return json.dumps({"object": "list", "data": [{"id": i} for i in ids]})

    def test_model_served(self) -> None:
        result = dg.check_model_available(self._body(["alias-fast", "alias-eve"]), "blablador:alias-fast")
        assert result.ok
        assert "alias-fast" in result.detail

    def test_model_missing_warns_with_count(self) -> None:
        result = dg.check_model_available(self._body(["alias-fast"]), "blablador:nope")
        assert not result.ok
        assert result.warn
        assert "1 served" in result.detail

    def test_model_missing_suggests_similar(self) -> None:
        result = dg.check_model_available(self._body(["alias-fast", "alias-eve"]), "blablador:fast")
        assert "alias-fast" in result.detail

    def test_unparseable_body_warns(self) -> None:
        result = dg.check_model_available("<html>not json</html>", "blablador:x")
        assert result.warn and not result.ok


class TestCheckEndpoint:
    def test_missing_base_url_fails(self) -> None:
        assert not dg.check_endpoint(None).ok

    def test_reachable(self) -> None:
        result = dg.check_endpoint("https://api.example.de", opener=lambda u, t: "ok")
        assert result.ok
        assert result.detail.endswith("reachable")

    def test_opener_receives_v1_models_url(self) -> None:
        seen: list[str] = []
        dg.check_endpoint("https://api.example.de", opener=lambda u, t: seen.append(u) or "ok")
        assert seen == ["https://api.example.de/v1/models"]

    def test_http_error_reports_code(self) -> None:
        def boom(url: str, timeout: float) -> str:
            raise _http_error(503)

        result = dg.check_endpoint("https://api.example.de", opener=boom)
        assert not result.ok
        assert "HTTP 503" in result.detail

    def test_network_error_reports_brief(self) -> None:
        def boom(url: str, timeout: float) -> str:
            raise OSError("no route to host")

        result = dg.check_endpoint("https://api.example.de", opener=boom)
        assert not result.ok
        assert "unreachable" in result.detail


class TestCheckChat:
    def test_missing_credentials_skips_as_warn(self) -> None:
        result = dg.check_chat(None, "https://api.example.de", "blablador:x")
        assert result.warn and result.ok

    def test_success(self) -> None:
        poster = lambda url, headers, body, timeout: "ok"  # noqa: E731
        result = dg.check_chat("sk-key", "https://api.example.de", "blablador:x", poster=poster)
        assert result.ok
        assert result.detail == "model responded"

    def test_probe_body_uses_bare_model_id(self) -> None:
        seen: dict[str, str] = {}

        def poster(url: str, headers: dict[str, str], body: str, timeout: float) -> str:
            seen["url"], seen["body"], seen["auth"] = url, body, headers["Authorization"]
            return "ok"

        dg.check_chat("sk-key", "https://api.example.de/v1", "blablador:alias-fast", poster=poster)
        assert seen["url"] == "https://api.example.de/v1/chat/completions"
        assert '"model": "alias-fast"' in seen["body"]
        assert seen["auth"] == "Bearer sk-key"

    @pytest.mark.parametrize("code", [401, 403])
    def test_auth_errors_fail(self, code: int) -> None:
        def poster(url: str, headers: dict[str, str], body: str, timeout: float) -> str:
            raise _http_error(code)

        result = dg.check_chat("sk-key", "https://api.example.de", "blablador:x", poster=poster)
        assert not result.ok and not result.warn
        assert "check API key" in result.detail

    def test_unknown_model_404_warns(self) -> None:
        def poster(url: str, headers: dict[str, str], body: str, timeout: float) -> str:
            raise _http_error(404)

        result = dg.check_chat("sk-key", "https://api.example.de", "blablador:x", poster=poster)
        assert not result.ok and result.warn
        assert "404" in result.detail

    def test_other_http_codes_pass_as_contact(self) -> None:
        def poster(url: str, headers: dict[str, str], body: str, timeout: float) -> str:
            raise _http_error(400)

        result = dg.check_chat("sk-key", "https://api.example.de", "blablador:x", poster=poster)
        assert result.ok
        assert "HTTP 400" in result.detail

    def test_network_error_fails(self) -> None:
        def poster(url: str, headers: dict[str, str], body: str, timeout: float) -> str:
            raise OSError("connection refused")

        result = dg.check_chat("sk-key", "https://api.example.de", "blablador:x", poster=poster)
        assert not result.ok
