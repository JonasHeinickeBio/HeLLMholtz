"""Regression tests for the blablador_models.py writer used by `hellm models sync`."""

from pathlib import Path

from hellmholtz.providers.blablador_config import _update_models_file

BASE = '''\
from dataclasses import dataclass


@dataclass
class BlabladorModel:
    id: str = ""
    name: str = ""
    description: str = ""
    source: str = ""
    max_context_tokens: int = 0
    available: bool = True


KNOWN_MODELS = [
    BlabladorModel(
        id="01",
        name="Alpha",
        description="a",
        max_context_tokens=1000,
    ),
    BlabladorModel(
        id="",
        name="alias-beta",
        max_context_tokens=2000,
        available=False,
    ),
    BlabladorModel(
        id="03",
        name="Gamma",
        max_context_tokens=3000,
    ),
]
'''


def _run(tmp_path: Path, api: list[dict], removed: list[str]):
    path = tmp_path / "models.py"
    path.write_text(BASE)
    result = _update_models_file(path, api, removed)
    namespace: dict = {}
    exec(compile(path.read_text(), str(path), "exec"), namespace)  # must be valid Python
    models = {m.name: m for m in namespace["KNOWN_MODELS"]}
    return result, models, path.read_text()


def test_removed_models_are_marked_once_even_when_adjacent(tmp_path):
    result, models, text = _run(tmp_path, [{"id": "alias-beta"}], ["Alpha", "Gamma"])

    assert result["success"]
    assert not models["Alpha"].available and not models["Gamma"].available
    assert text.count("available=False") == 2  # Alpha and Gamma; beta was re-enabled


def test_already_unavailable_model_is_not_marked_twice(tmp_path):
    _, _, text = _run(tmp_path, [], ["alias-beta"])

    assert text.count("available=False") == 1


def test_returning_model_is_reenabled(tmp_path):
    _, models, _ = _run(tmp_path, [{"id": "alias-beta"}], [])

    assert models["alias-beta"].available


def test_new_entries_use_api_context_window_and_bare_alias_ids(tmp_path):
    api = [
        {"id": "alias-new", "max_model_len": 150000},
        {"id": "10 - A good european model - Mistral 4 Large", "max_model_len": 262144},
    ]
    _, models, _ = _run(tmp_path, api, [])

    assert models["alias-new"].max_context_tokens == 150000
    assert models["alias-new"].id == ""  # so api_id resolves to the bare name
    tagline = models["A good european model"]
    assert tagline.id == "10 - A good european model - Mistral 4 Large"
    assert tagline.max_context_tokens == 262144
