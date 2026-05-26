import logging
import re

from hellmholtz.core.config import get_settings
from hellmholtz.providers.blablador_config import KNOWN_MODELS, BlabladorModel

logger = logging.getLogger(__name__)


def _build_known_model_indexes() -> tuple[
    dict[str, list[BlabladorModel]], dict[str, BlabladorModel]
]:
    """Build lookups for matching raw API IDs to known models."""
    known_by_id: dict[str, list[BlabladorModel]] = {}
    for known_model in KNOWN_MODELS:
        known_by_id.setdefault(known_model.id, []).append(known_model)

    known_by_name = {known_model.name: known_model for known_model in KNOWN_MODELS}
    return known_by_id, known_by_name


def _parse_raw_model_id(raw_model: str) -> BlabladorModel:
    """Parse a raw API model ID into a BlabladorModel instance."""
    match = re.match(r"^(\d+)\s-\s(.*?)\s-\s(.*)$", raw_model)
    if match:
        return BlabladorModel(
            id=match.group(1),
            name=match.group(2),
            description=match.group(3),
            source="Blablador",
            original_api_id=raw_model,
        )

    match = re.match(r"^(\d+)\s-\s(.*)$", raw_model)
    if match:
        return BlabladorModel(
            id=match.group(1),
            name=match.group(2),
            description="",
            source="Blablador",
            original_api_id=raw_model,
        )

    return BlabladorModel(
        id=raw_model,
        name=raw_model,
        description="",
        source="Blablador",
        original_api_id=raw_model,
    )


def _find_best_known_match(
    model_obj: BlabladorModel,
    known_by_id: dict[str, list[BlabladorModel]],
    known_by_name: dict[str, BlabladorModel],
) -> BlabladorModel | None:
    """Find the best known-model match for a parsed model."""
    if model_obj.id in known_by_id:
        candidates = known_by_id[model_obj.id]
        for candidate in candidates:
            if candidate.name == model_obj.name:
                return candidate
            if candidate.name in model_obj.name or model_obj.name in candidate.name:
                return candidate
        if len(candidates) == 1:
            return candidates[0]

    return known_by_name.get(model_obj.name)


def _enrich_model_from_known_data(
    model_obj: BlabladorModel,
    known_by_id: dict[str, list[BlabladorModel]],
    known_by_name: dict[str, BlabladorModel],
) -> BlabladorModel:
    """Merge known model metadata into a parsed API model."""
    best_match = _find_best_known_match(model_obj, known_by_id, known_by_name)
    if not best_match:
        return model_obj

    if not model_obj.description:
        model_obj.description = best_match.description
    model_obj.name = best_match.name
    model_obj.alias = best_match.alias
    model_obj.source = best_match.source
    return model_obj


def parse_api_model_ids(raw_model_ids: list[str]) -> list[BlabladorModel]:
    """Parse raw API model IDs into enriched BlabladorModel objects.

    This utility is shared by different code paths (provider, monitor) to keep
    model-ID normalization consistent across availability checks.
    """
    parsed_models: list[BlabladorModel] = []
    known_by_id, known_by_name = _build_known_model_indexes()

    for raw_model in raw_model_ids:
        model_obj = _parse_raw_model_id(raw_model)
        parsed_models.append(_enrich_model_from_known_data(model_obj, known_by_id, known_by_name))

    return parsed_models


def list_models() -> list[BlabladorModel]:  # noqa: C901
    """List available models from Blablador (OpenAI-compatible)."""
    import httpx

    settings = get_settings()

    if not settings.blablador_api_key or not settings.blablador_base_url:
        raise ValueError("Blablador API key and Base URL must be set in .env")

    headers = {
        "Authorization": f"Bearer {settings.blablador_api_key}",
        "Content-Type": "application/json",
    }

    url = f"{settings.blablador_base_url.rstrip('/')}/models"
    logger.debug(f"Fetching models from {url}")

    try:
        response = httpx.get(url, headers=headers, timeout=settings.timeout_seconds)
        response.raise_for_status()
        data = response.json()

        raw_models = [model["id"] for model in data.get("data", []) if isinstance(model, dict)]
        parsed_models = parse_api_model_ids(raw_models)

        logger.info(f"Successfully fetched {len(parsed_models)} models from Blablador")
        return parsed_models

    except Exception as e:
        logger.error(f"Failed to fetch models: {e}")
        raise RuntimeError(f"Failed to fetch models from Blablador: {e}") from e
