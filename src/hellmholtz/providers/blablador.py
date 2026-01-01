import logging
import re

from hellmholtz.core.config import get_settings
from hellmholtz.providers.blablador_config import KNOWN_MODELS, BlabladorModel

logger = logging.getLogger(__name__)


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

        raw_models = [model["id"] for model in data.get("data", [])]
        parsed_models = []

        # Create lookups for known models
        known_by_id: dict[str, list[BlabladorModel]] = {}
        for m in KNOWN_MODELS:
            if m.id not in known_by_id:
                known_by_id[m.id] = []
            known_by_id[m.id].append(m)

        known_by_name = {m.name: m for m in KNOWN_MODELS}

        for raw_model in raw_models:
            model_obj = None

            # Try to parse "ID - Name - Description" format
            match = re.match(r"^(\d+)\s-\s(.*?)\s-\s(.*)$", raw_model)
            if match:
                model_obj = BlabladorModel(
                    id=match.group(1),
                    name=match.group(2),
                    description=match.group(3),
                    source="Blablador",
                    original_api_id=raw_model,
                )
            else:
                # Pattern: Start with digits, space hyphen space, anything (no description)
                match = re.match(r"^(\d+)\s-\s(.*)$", raw_model)
                if match:
                    model_obj = BlabladorModel(
                        id=match.group(1),
                        name=match.group(2),
                        description="",
                        source="Blablador",
                        original_api_id=raw_model,
                    )
                else:
                    # Fallback: Use the whole string as ID and Name
                    model_obj = BlabladorModel(
                        id=raw_model,
                        name=raw_model,
                        description="",
                        source="Blablador",
                        original_api_id=raw_model,
                    )

            # Enrich with known data
            # Try matching by ID first
            if model_obj.id in known_by_id:
                candidates = known_by_id[model_obj.id]
                best_match = None

                # Try to find a candidate that matches the name
                for candidate in candidates:
                    # Check if names are similar or contained
                    if candidate.name == model_obj.name:
                        best_match = candidate
                        break
                    if candidate.name in model_obj.name or model_obj.name in candidate.name:
                        best_match = candidate
                        break

                # If no name match but only one candidate, assume it's that one
                if not best_match and len(candidates) == 1:
                    best_match = candidates[0]

                if best_match:
                    if not model_obj.description:
                        model_obj.description = best_match.description
                    # Always prefer known name to ensure clean display
                    model_obj.name = best_match.name
                    model_obj.alias = best_match.alias
                    model_obj.source = best_match.source

            # Try matching by Name (if ID didn't match or wasn't present)
            elif model_obj.name in known_by_name:
                known = known_by_name[model_obj.name]
                if not model_obj.description:
                    model_obj.description = known.description
                model_obj.alias = known.alias
                model_obj.source = known.source

            parsed_models.append(model_obj)

        logger.info(f"Successfully fetched {len(parsed_models)} models from Blablador")
        return parsed_models

    except Exception as e:
        logger.error(f"Failed to fetch models: {e}")
        raise RuntimeError(f"Failed to fetch models from Blablador: {e}") from e
