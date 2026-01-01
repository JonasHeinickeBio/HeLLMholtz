from pathlib import Path
import sys

# Ensure src is in python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from hellmholtz.benchmark import run_benchmarks
from hellmholtz.core.prompts import Message, Prompt
from hellmholtz.providers.blablador import list_models
from hellmholtz.reporting import summarize_results


def main() -> None:
    print("Fetching Blablador models...")
    try:
        models = list_models()
    except Exception as e:
        print(f"Error fetching models: {e}")
        sys.exit(1)

    if not models:
        print("No models found.")
        sys.exit(1)

    # Prepare model list for hellmholtz
    bench_models = []
    for m in models:
        # Use name if available to ensure correct resolution in BlabladorProvider,
        # especially for models with duplicate IDs (like '1').
        identifier = m.name if m.name else m.id
        bench_models.append(f"blablador:{identifier}")

    print(f"Found {len(bench_models)} models: {', '.join(bench_models)}")

    # Load prompts
    prompts_file = Path("prompts.txt")
    if not prompts_file.exists():
        # Try looking in parent dir if running from scripts/
        prompts_file = Path("../prompts.txt")
        if not prompts_file.exists():
            print("prompts.txt not found.")
            sys.exit(1)

    with open(prompts_file) as f:
        prompt_strings = [line.strip() for line in f if line.strip()]

    # Convert strings to Prompt objects
    prompts = []
    for i, prompt_text in enumerate(prompt_strings):
        prompts.append(
            Prompt(
                id=f"prompt_{i + 1}",
                category="custom",
                messages=[Message(role="user", content=prompt_text)],
                description=f"Custom prompt {i + 1}",
            )
        )

    print(f"Running benchmarks with {len(prompts)} prompts...")

    # Run benchmarks
    # We'll use a try-except block to ensure we print what we have even if it crashes mid-way
    try:
        results = run_benchmarks(bench_models, prompts, repeat=1)
        print("\nBenchmark Results Summary:")
        print(summarize_results(results))

        # Also print path to results
        print("\nDetailed results saved to results/ directory.")

    except Exception as e:
        print(f"\nAn error occurred during benchmarking: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
