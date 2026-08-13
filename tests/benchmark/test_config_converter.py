"""Comprehensive tests for config_converter package."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from config_converter import (
    convert_file,
    get_secret,
    json_to_yaml,
    load_dotenv_files,
    mask_secrets,
    print_validation_report,
    resolve_env_refs,
    validate_roundtrip,
    yaml_to_json,
)
from config_converter.convert import _is_file_path
from config_converter.secrets import SECRET_ENV_MAP, SECRET_PATTERNS, _is_secret_key
from config_converter.validate import _compare_scalars, _deep_compare


# ---------------------------------------------------------------------------
# __init__.py – export checks
# ---------------------------------------------------------------------------

class TestExports:
    def test_all_public_names_importable(self):
        import config_converter as cc

        for name in [
            "convert_file",
            "json_to_yaml",
            "yaml_to_json",
            "mask_secrets",
            "resolve_env_refs",
            "load_dotenv_files",
            "get_secret",
            "validate_roundtrip",
            "print_validation_report",
        ]:
            assert hasattr(cc, name)

    def test_all_matches_actual_exports(self):
        import config_converter as cc

        for name in cc.__all__:
            assert hasattr(cc, name)


# ---------------------------------------------------------------------------
# convert.py – _is_file_path
# ---------------------------------------------------------------------------

class TestIsFilePath:
    def test_path_object_returns_true(self, tmp_path: Path):
        f = tmp_path / "a.txt"
        f.write_text("x")
        assert _is_file_path(f) is True

    def test_string_existing_file(self, tmp_path: Path):
        f = tmp_path / "b.txt"
        f.write_text("x")
        assert _is_file_path(str(f)) is True

    def test_string_nonexistent_file(self):
        assert _is_file_path("/no/such/file.txt") is False

    def test_plain_string_not_path(self):
        assert _is_file_path("hello world") is False

    def test_non_string_non_path(self):
        assert _is_file_path(123) is False  # type: ignore[arg-type]

    def test_directory_not_file(self, tmp_path: Path):
        d = tmp_path / "dir"
        d.mkdir()
        assert _is_file_path(str(d)) is False


# ---------------------------------------------------------------------------
# convert.py – json_to_yaml
# ---------------------------------------------------------------------------

class TestJsonToYaml:
    def test_from_string(self):
        result = json_to_yaml('{"a": 1, "b": [1, 2]}')
        parsed = yaml.safe_load(result)
        assert parsed == {"a": 1, "b": [1, 2]}

    def test_from_file(self, tmp_path: Path):
        f = tmp_path / "in.json"
        f.write_text('{"key": "value"}')
        result = json_to_yaml(f)
        assert "key: value" in result

    def test_output_path(self, tmp_path: Path):
        out = tmp_path / "sub" / "out.yaml"
        json_to_yaml('{"x": 1}', output_path=out)
        assert out.exists()
        assert yaml.safe_load(out.read_text()) == {"x": 1}

    def test_sort_keys(self):
        result = json_to_yaml('{"z": 1, "a": 2}', sort_keys=True)
        lines = result.strip().split("\n")
        assert lines[0].startswith("a:")
        assert lines[1].startswith("z:")

    def test_mask_api_keys(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "")
        data = {"openai_api_key": "sk-secret123", "name": "test"}
        result = json_to_yaml(json.dumps(data), mask_api_keys=True)
        parsed = yaml.safe_load(result)
        assert parsed["openai_api_key"] == "${OPENAI_API_KEY}"
        assert parsed["name"] == "test"

    def test_env_files(self, tmp_path: Path):
        env = tmp_path / ".env"
        env.write_text("TESTVAR=hello\n")
        data = {"val": "test"}
        result = json_to_yaml(json.dumps(data), env_files=[str(env)])
        parsed = yaml.safe_load(result)
        assert parsed == data

    def test_custom_indent(self):
        data = {"section": {"key": "value"}}
        result = json_to_yaml(json.dumps(data), indent=4)
        assert "\n    key:" in result

    def test_preserves_unicode(self):
        result = json_to_yaml('{"emoji": "\u2764"}')
        parsed = yaml.safe_load(result)
        assert parsed["emoji"] == "\u2764"


# ---------------------------------------------------------------------------
# convert.py – yaml_to_json
# ---------------------------------------------------------------------------

class TestYamlToJson:
    def test_from_string(self):
        result = yaml_to_json("a: 1\nb:\n  - 2\n  - 3")
        parsed = json.loads(result)
        assert parsed == {"a": 1, "b": [2, 3]}

    def test_from_file(self, tmp_path: Path):
        f = tmp_path / "in.yaml"
        f.write_text("hello: world\n")
        result = yaml_to_json(f)
        parsed = json.loads(result)
        assert parsed == {"hello": "world"}

    def test_output_path(self, tmp_path: Path):
        out = tmp_path / "out.json"
        yaml_to_json("a: 1", output_path=str(out))
        assert out.exists()
        assert json.loads(out.read_text()) == {"a": 1}

    def test_resolve_env(self, monkeypatch):
        monkeypatch.setenv("MY_TOKEN", "resolved_value")
        data = "token: ${MY_TOKEN}\n"
        result = yaml_to_json(data, resolve_env=True)
        parsed = json.loads(result)
        assert parsed["token"] == "resolved_value"

    def test_resolve_env_missing_var(self):
        data = "val: ${NONEXISTENT_VAR_XYZ}\n"
        result = yaml_to_json(data, resolve_env=True)
        parsed = json.loads(result)
        assert parsed["val"] == "${NONEXISTENT_VAR_XYZ}"

    def test_env_files(self, tmp_path: Path):
        env = tmp_path / ".env"
        env.write_text("DOTVAR=fromfile\n")
        data = "v: ${DOTVAR}\n"
        result = yaml_to_json(data, resolve_env=True, env_files=[str(env)])
        parsed = json.loads(result)
        assert parsed["v"] == "fromfile"

    def test_custom_indent(self):
        result = yaml_to_json("a: 1", indent=4)
        parsed = json.loads(result)
        assert parsed == {"a": 1}


# ---------------------------------------------------------------------------
# convert.py – convert_file
# ---------------------------------------------------------------------------

class TestConvertFile:
    def test_json_to_yaml_auto_output(self, tmp_path: Path):
        f = tmp_path / "data.json"
        f.write_text('{"a": 1}')
        out = convert_file(f)
        assert out.suffix == ".yaml"
        assert yaml.safe_load(out.read_text()) == {"a": 1}

    def test_json_to_yaml_explicit_output(self, tmp_path: Path):
        f = tmp_path / "data.json"
        f.write_text('{"a": 1}')
        out = tmp_path / "custom.yaml"
        result = convert_file(f, out)
        assert result == out
        assert out.exists()

    def test_yaml_to_json_auto_output(self, tmp_path: Path):
        f = tmp_path / "data.yaml"
        f.write_text("a: 1\n")
        out = convert_file(f)
        assert out.suffix == ".json"
        assert json.loads(out.read_text()) == {"a": 1}

    def test_yaml_to_json_explicit_output(self, tmp_path: Path):
        f = tmp_path / "data.yml"
        f.write_text("a: 1\n")
        out = tmp_path / "custom.json"
        result = convert_file(f, out)
        assert result == out
        assert json.loads(out.read_text()) == {"a": 1}

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            convert_file("/no/such/file.json")

    def test_bad_extension_raises(self, tmp_path: Path):
        f = tmp_path / "data.csv"
        f.write_text("a,b")
        with pytest.raises(ValueError, match="Unsupported"):
            convert_file(f)

    def test_sort_keys(self, tmp_path: Path):
        f = tmp_path / "data.json"
        f.write_text('{"z": 1, "a": 2}')
        out = convert_file(f, sort_keys=True)
        text = out.read_text()
        assert text.index("a:") < text.index("z:")

    def test_mask_api_keys(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "")
        f = tmp_path / "data.json"
        f.write_text('{"openai_api_key": "sk-test"}')
        out = convert_file(f, mask_api_keys=True)
        parsed = yaml.safe_load(out.read_text())
        assert parsed["openai_api_key"] == "${OPENAI_API_KEY}"

    def test_resolve_env(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("RESOLVE_ME", "done")
        f = tmp_path / "data.yaml"
        f.write_text("val: ${RESOLVE_ME}\n")
        out = convert_file(f, resolve_env=True)
        parsed = json.loads(out.read_text())
        assert parsed["val"] == "done"

    def test_env_files_passed_through(self, tmp_path: Path, monkeypatch):
        env = tmp_path / ".env"
        env.write_text("CONV_ENV=from_env\n")
        f = tmp_path / "data.yaml"
        f.write_text("v: ${CONV_ENV}\n")
        out = convert_file(f, resolve_env=True, env_files=[str(env)])
        parsed = json.loads(out.read_text())
        assert parsed["v"] == "from_env"


# ---------------------------------------------------------------------------
# secrets.py – _is_secret_key
# ---------------------------------------------------------------------------

class TestIsSecretKey:
    @pytest.mark.parametrize(
        "key",
        [
            "api_key",
            "apiKey",
            "api-key",
            "API_KEY",
            "my_api_key",
            "secret",
            "SECRET",
            "my_secret",
            "token",
            "access_token",
            "TOKEN",
            "password",
            "user_password",
            "PASSWORD",
            "credential",
            "db_credential",
            "CREDENTIAL",
        ],
    )
    def test_secret_keys_detected(self, key: str):
        assert _is_secret_key(key) is True

    @pytest.mark.parametrize(
        "key",
        [
            "name",
            "host",
            "port",
            "debug",
            "enabled",
            "url",
            "api_url",
            "timeout",
        ],
    )
    def test_normal_keys_not_detected(self, key: str):
        assert _is_secret_key(key) is False


# ---------------------------------------------------------------------------
# secrets.py – SECRET_PATTERNS / SECRET_ENV_MAP constants
# ---------------------------------------------------------------------------

class TestSecretConstants:
    def test_secret_patterns_is_list_of_strings(self):
        assert isinstance(SECRET_PATTERNS, list)
        assert all(isinstance(p, str) for p in SECRET_PATTERNS)

    def test_secret_env_map_is_dict(self):
        assert isinstance(SECRET_ENV_MAP, dict)
        assert "api_key" in SECRET_ENV_MAP
        assert "OPENAI_API_KEY" in SECRET_ENV_MAP


# ---------------------------------------------------------------------------
# secrets.py – mask_secrets
# ---------------------------------------------------------------------------

class TestMaskSecrets:
    def test_mask_known_key_openai(self):
        data = {"OPENAI_API_KEY": "sk-abc", "name": "test"}
        result = mask_secrets(data)
        assert result["OPENAI_API_KEY"] == "${OPENAI_API_KEY}"
        assert result["name"] == "test"

    def test_mask_known_key_api_key(self):
        data = {"api_key": "secret123"}
        result = mask_secrets(data)
        assert result["api_key"] == "${BLABLADOR_API_KEY}"

    def test_mask_unknown_secret_key(self):
        data = {"my_custom_token": "tok123"}
        result = mask_secrets(data)
        assert result["my_custom_token"] == "${MY_CUSTOM_TOKEN}"

    def test_mask_preserves_non_secret_values(self):
        data = {"port": 8080, "host": "localhost"}
        assert mask_secrets(data) == {"port": 8080, "host": "localhost"}

    def test_mask_empty_string_not_masked(self):
        data = {"api_key": ""}
        result = mask_secrets(data)
        assert result["api_key"] == ""

    def test_mask_nested_dict(self):
        data = {"server": {"api_key": "secret123", "port": 80}}
        result = mask_secrets(data)
        assert result["server"]["api_key"] == "${BLABLADOR_API_KEY}"
        assert result["server"]["port"] == 80

    def test_mask_list_of_dicts(self):
        data = [{"password": "pw1"}, {"password": "pw2"}]
        result = mask_secrets(data)
        assert result[0]["password"] == "${PASSWORD}"
        assert result[1]["password"] == "${PASSWORD}"

    def test_mask_scalar_passthrough(self):
        assert mask_secrets("just a string") == "just a string"
        assert mask_secrets(42) == 42
        assert mask_secrets(None) is None

    def test_mask_list_of_scalars(self):
        data = [1, "two", None]
        assert mask_secrets(data) == [1, "two", None]

    def test_mask_custom_env_map(self):
        data = {"secret": "val"}
        custom = {"secret": "CUSTOM_VAR"}
        result = mask_secrets(data, env_map=custom)
        assert result["secret"] == "${CUSTOM_VAR}"

    def test_mask_non_string_secret_value_not_masked(self):
        data = {"api_key": 12345}
        result = mask_secrets(data)
        assert result["api_key"] == 12345

    def test_mask_deeply_nested(self):
        data = {"a": {"b": {"c": {"token": "xyz"}}}}
        result = mask_secrets(data)
        assert result["a"]["b"]["c"]["token"] == "${TOKEN}"

    def test_mask_list_in_nested_dict(self):
        data = {"items": [{"password": "p1"}, "plain", {"secret": "s1"}]}
        result = mask_secrets(data)
        assert result["items"][0]["password"] == "${PASSWORD}"
        assert result["items"][1] == "plain"
        assert result["items"][2]["secret"] == "${SECRET}"


# ---------------------------------------------------------------------------
# secrets.py – resolve_env_refs
# ---------------------------------------------------------------------------

class TestResolveEnvRefs:
    def test_full_var_reference(self, monkeypatch):
        monkeypatch.setenv("FULL_VAR", "resolved")
        result = resolve_env_refs("${FULL_VAR}")
        assert result == "resolved"

    def test_missing_var_keeps_original(self):
        result = resolve_env_refs("${NOPE_NOPE_NOPE_999}")
        assert result == "${NOPE_NOPE_NOPE_999}"

    def test_inline_reference(self, monkeypatch):
        monkeypatch.setenv("HOST", "localhost")
        result = resolve_env_refs("http://${HOST}:8080")
        assert result == "http://localhost:8080"

    def test_multiple_inline_refs(self, monkeypatch):
        monkeypatch.setenv("A", "alpha")
        monkeypatch.setenv("B", "beta")
        result = resolve_env_refs("${A}-${B}")
        assert result == "alpha-beta"

    def test_missing_inline_var_keeps_ref(self, monkeypatch):
        monkeypatch.setenv("A", "alpha")
        result = resolve_env_refs("${A}-${MISSING}")
        assert result == "alpha-${MISSING}"

    def test_dict_recursion(self, monkeypatch):
        monkeypatch.setenv("DB_PASS", "secret123")
        data = {"db": {"password": "${DB_PASS}", "host": "localhost"}}
        result = resolve_env_refs(data)
        assert result["db"]["password"] == "secret123"
        assert result["db"]["host"] == "localhost"

    def test_list_recursion(self, monkeypatch):
        monkeypatch.setenv("VAL", "x")
        data = ["${VAL}", "plain", 42]
        result = resolve_env_refs(data)
        assert result == ["x", "plain", 42]

    def test_non_string_passthrough(self):
        assert resolve_env_refs(42) == 42
        assert resolve_env_refs(None) is None
        assert resolve_env_refs(True) is True

    def test_nested_list_of_dicts(self, monkeypatch):
        monkeypatch.setenv("K", "v")
        data = [{"key": "${K}"}, {"key": "literal"}]
        result = resolve_env_refs(data)
        assert result[0]["key"] == "v"
        assert result[1]["key"] == "literal"


# ---------------------------------------------------------------------------
# secrets.py – load_dotenv_files
# ---------------------------------------------------------------------------

class TestLoadDotenvFiles:
    def test_load_existing_file(self, tmp_path: Path, monkeypatch):
        env = tmp_path / ".env"
        env.write_text("LOADED_VAR=from_dotenv\n")
        load_dotenv_files(str(env))
        assert os.environ.get("LOADED_VAR") == "from_dotenv"
        monkeypatch.delenv("LOADED_VAR", raising=False)

    def test_missing_file_ignored(self):
        load_dotenv_files("/no/such/.env")

    def test_multiple_files(self, tmp_path: Path, monkeypatch):
        e1 = tmp_path / "a.env"
        e1.write_text("VAR_A=1\n")
        e2 = tmp_path / "b.env"
        e2.write_text("VAR_B=2\n")
        load_dotenv_files(str(e1), str(e2))
        assert os.environ.get("VAR_A") == "1"
        assert os.environ.get("VAR_B") == "2"
        monkeypatch.delenv("VAR_A", raising=False)
        monkeypatch.delenv("VAR_B", raising=False)

    def test_override_false_does_not_clobber(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("EXISTING", "original")
        env = tmp_path / ".env"
        env.write_text("EXISTING=overwritten\n")
        load_dotenv_files(str(env), override=False)
        assert os.environ["EXISTING"] == "original"
        monkeypatch.delenv("EXISTING", raising=False)


# ---------------------------------------------------------------------------
# secrets.py – get_secret
# ---------------------------------------------------------------------------

class TestGetSecret:
    def test_explicit_env_var(self, monkeypatch):
        monkeypatch.setenv("EXPLICIT", "val1")
        assert get_secret("any_key", env_var="EXPLICIT") == "val1"

    def test_auto_detection_known_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-123")
        assert get_secret("OPENAI_API_KEY") == "sk-123"
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def test_auto_detection_api_key(self, monkeypatch):
        monkeypatch.setenv("BLABLADOR_API_KEY", "blabla")
        assert get_secret("api_key") == "blabla"
        monkeypatch.delenv("BLABLADOR_API_KEY", raising=False)

    def test_auto_detection_unknown_key_uppered(self, monkeypatch):
        monkeypatch.setenv("MY_FOO", "bar")
        assert get_secret("my_foo") == "bar"
        monkeypatch.delenv("MY_FOO", raising=False)

    def test_missing_returns_default(self):
        assert get_secret("nonexistent_key_xyz", default="fallback") == "fallback"

    def test_missing_no_default_returns_none(self):
        assert get_secret("nonexistent_key_xyz") is None

    def test_explicit_env_var_overrides_auto(self, monkeypatch):
        monkeypatch.setenv("CUSTOM_VAR", "custom_val")
        monkeypatch.setenv("BLABLADOR_API_KEY", "auto_val")
        result = get_secret("api_key", env_var="CUSTOM_VAR")
        assert result == "custom_val"
        monkeypatch.delenv("CUSTOM_VAR", raising=False)
        monkeypatch.delenv("BLABLADOR_API_KEY", raising=False)


# ---------------------------------------------------------------------------
# validate.py – _compare_scalars
# ---------------------------------------------------------------------------

class TestCompareScalars:
    def test_both_none(self):
        assert _compare_scalars(None, None) is True

    def test_one_none(self):
        assert _compare_scalars(None, 1) is False
        assert _compare_scalars(1, None) is False

    def test_both_bool(self):
        assert _compare_scalars(True, True) is True
        assert _compare_scalars(True, False) is False

    def test_bool_and_str(self):
        assert _compare_scalars(True, "True") is True
        assert _compare_scalars("true", False) is False

    def test_str_and_bool(self):
        assert _compare_scalars("True", True) is True

    def test_both_str(self):
        assert _compare_scalars("abc", "abc") is True
        assert _compare_scalars("abc", "def") is False

    def test_both_int(self):
        assert _compare_scalars(5, 5) is True
        assert _compare_scalars(5, 6) is False

    def test_both_float(self):
        assert _compare_scalars(1.0, 1.0 + 1e-11) is True
        assert _compare_scalars(1.0, 2.0) is False

    def test_int_float_mixed(self):
        assert _compare_scalars(1, 1.0) is True

    def test_int_and_str_numeric(self):
        assert _compare_scalars(42, "42.0") is True

    def test_str_and_int_numeric(self):
        assert _compare_scalars("3.14", 3.14) is True

    def test_str_non_numeric(self):
        assert _compare_scalars(42, "abc") is False

    def test_int_str_non_numeric(self):
        assert _compare_scalars("abc", 42) is False

    def test_non_scalar_returns_none(self):
        assert _compare_scalars([1], [1]) is None
        assert _compare_scalars({"a": 1}, {"a": 1}) is None

    def test_none_vs_bool(self):
        assert _compare_scalars(None, True) is False

    def test_bool_vs_int(self):
        # bool is a subclass of int, so this hits the numeric branch
        assert _compare_scalars(True, 2) is False


# ---------------------------------------------------------------------------
# validate.py – _deep_compare
# ---------------------------------------------------------------------------

class TestDeepCompare:
    def test_equal_dicts(self):
        assert _deep_compare({"a": 1, "b": 2}, {"a": 1, "b": 2}) is True

    def test_different_dicts(self):
        assert _deep_compare({"a": 1}, {"a": 2}) is False

    def test_different_keys(self):
        assert _deep_compare({"a": 1}, {"b": 1}) is False

    def test_equal_lists(self):
        assert _deep_compare([1, 2, 3], [1, 2, 3]) is True

    def test_different_lists(self):
        assert _deep_compare([1, 2], [1, 3]) is False

    def test_different_length_lists(self):
        assert _deep_compare([1], [1, 2]) is False

    def test_nested_structures(self):
        a = {"outer": {"inner": [1, 2, {"deep": True}]}}
        b = {"outer": {"inner": [1, 2, {"deep": True}]}}
        assert _deep_compare(a, b) is True

    def test_nested_mismatch(self):
        a = {"outer": {"inner": [1]}}
        b = {"outer": {"inner": [2]}}
        assert _deep_compare(a, b) is False

    def test_scalar_comparison(self):
        assert _deep_compare(42, 42) is True
        assert _deep_compare(42, 99) is False

    def test_bool_int_yaml_coercion(self):
        assert _deep_compare(True, 1) is True
        assert _deep_compare(False, 0) is True

    def test_none_comparison(self):
        assert _deep_compare(None, None) is True
        assert _deep_compare(None, 1) is False

    def test_mixed_nested_with_booleans(self):
        a = {"flag": True, "count": 1}
        b = {"flag": True, "count": 1}
        assert _deep_compare(a, b) is True

    def test_fallback_direct_equality(self):
        # Non-scalar, non-dict, non-list: tuples
        assert _deep_compare((1, 2), (1, 2)) is True
        assert _deep_compare((1, 2), (1, 3)) is False

    def test_dict_vs_list_fallback(self):
        assert _deep_compare({"a": 1}, [1]) is False


# ---------------------------------------------------------------------------
# validate.py – validate_roundtrip
# ---------------------------------------------------------------------------

class TestValidateRoundtrip:
    def test_json_roundtrip(self, tmp_path: Path):
        f = tmp_path / "data.json"
        f.write_text(json.dumps({"a": 1, "b": [2, 3]}))
        result = validate_roundtrip(f)
        assert result["success"] is True
        assert result["input_format"] == "json"
        assert result["data_match"] is True
        assert result["errors"] == []

    def test_yaml_roundtrip(self, tmp_path: Path):
        f = tmp_path / "data.yaml"
        f.write_text("a: 1\nb:\n  - 2\n  - 3\n")
        result = validate_roundtrip(f)
        assert result["success"] is True
        assert result["input_format"] == "yaml"
        assert result["data_match"] is True

    def test_yml_extension(self, tmp_path: Path):
        f = tmp_path / "data.yml"
        f.write_text("x: 1\n")
        result = validate_roundtrip(f)
        assert result["input_format"] == "yaml"

    def test_missing_file(self):
        result = validate_roundtrip("/no/such/file.json")
        assert result["success"] is False
        assert "not found" in result["errors"][0]

    def test_bad_extension(self, tmp_path: Path):
        f = tmp_path / "data.csv"
        f.write_text("a,b")
        result = validate_roundtrip(f)
        assert result["success"] is False
        assert "Unsupported" in result["errors"][0]

    def test_explicit_output_format(self, tmp_path: Path):
        f = tmp_path / "data.json"
        f.write_text('{"x": 1}')
        result = validate_roundtrip(f, output_format="yaml")
        assert result["output_format"] == "yaml"

    def test_complex_data(self, tmp_path: Path):
        data = {
            "string": "hello",
            "number": 42,
            "float": 3.14,
            "bool": True,
            "null": None,
            "list": [1, 2, 3],
            "nested": {"key": "value"},
        }
        f = tmp_path / "complex.json"
        f.write_text(json.dumps(data))
        result = validate_roundtrip(f)
        assert result["success"] is True
        assert result["data_match"] is True

    def test_auto_output_format_json_input(self, tmp_path: Path):
        f = tmp_path / "test.json"
        f.write_text('{"auto": true}')
        result = validate_roundtrip(f, output_format="auto")
        assert result["output_format"] == "yaml"

    def test_auto_output_format_yaml_input(self, tmp_path: Path):
        f = tmp_path / "test.yaml"
        f.write_text("auto: true\n")
        result = validate_roundtrip(f, output_format="auto")
        assert result["output_format"] == "json"

    def test_corrupt_json_triggers_error(self, tmp_path: Path):
        f = tmp_path / "bad.json"
        f.write_text("{invalid json content!!")
        result = validate_roundtrip(f)
        assert result["success"] is False
        assert any("Conversion error" in e for e in result["errors"])


# ---------------------------------------------------------------------------
# validate.py – print_validation_report
# ---------------------------------------------------------------------------

class TestPrintValidationReport:
    def test_success_report(self):
        results = {
            "success": True,
            "input_format": "json",
            "output_format": "yaml",
            "data_match": True,
            "errors": [],
        }
        report = print_validation_report(results)
        assert "Round-Trip Validation Report" in report
        assert "json" in report
        assert "yaml" in report
        assert "All checks passed" in report

    def test_error_report(self):
        results = {
            "success": False,
            "input_format": "yaml",
            "output_format": "json",
            "data_match": False,
            "errors": ["Data structures differ after roundtrip"],
        }
        report = print_validation_report(results)
        assert "Errors:" in report
        assert "Data structures differ" in report

    def test_report_contains_delimiters(self):
        results = {
            "success": True,
            "input_format": "json",
            "output_format": "yaml",
            "data_match": True,
            "errors": [],
        }
        report = print_validation_report(results)
        assert report.startswith("=" * 50)
        assert report.endswith("=" * 50)

    def test_multiple_errors(self):
        results = {
            "success": False,
            "input_format": "unknown",
            "output_format": "auto",
            "data_match": False,
            "errors": ["error one", "error two"],
        }
        report = print_validation_report(results)
        assert "error one" in report
        assert "error two" in report

    def test_report_success_has_checkmark(self):
        results = {
            "success": True,
            "input_format": "json",
            "output_format": "yaml",
            "data_match": True,
            "errors": [],
        }
        report = print_validation_report(results)
        assert "\u2713" in report

    def test_report_error_has_cross(self):
        results = {
            "success": False,
            "input_format": "yaml",
            "output_format": "json",
            "data_match": False,
            "errors": ["bad"],
        }
        report = print_validation_report(results)
        assert "\u2717" in report
