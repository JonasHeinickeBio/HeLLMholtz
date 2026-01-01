from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / "src"))

from hellmholtz.providers.blablador import list_models
from hellmholtz.providers.blablador_config import KNOWN_MODELS


def main() -> None:
    print("Fetching API models...")
    api_models = list_models()

    print("\nComparing with KNOWN_MODELS:")
    for km in KNOWN_MODELS:
        print(f"\nChecking KNOWN_MODEL: {km.name}")
        # Reconstruct ID
        km_id = km.api_id  # This uses the full string logic now
        print(f"  KNOWN ID: '{km_id}'")

        # Find matching API model
        found = False
        for am in api_models:
            if am.name == km.name:
                found = True
                print(f"  API ID:   '{am.original_api_id}'")
                if km_id == am.original_api_id:
                    print("  MATCH: YES")
                else:
                    print("  MATCH: NO")
                    # Show diff
                    import difflib

                    for diff in difflib.ndiff([km_id], [am.original_api_id or ""]):
                        print(f"    {diff}")
                break
        if not found:
            print("  MATCH: NOT FOUND IN API")


if __name__ == "__main__":
    main()
