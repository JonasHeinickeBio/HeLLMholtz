import logging
import sys

try:
    import lm_eval
    from lm_eval import simple_evaluate
except ImportError:
    lm_eval = None


logger = logging.getLogger(__name__)


def run_lm_eval(
    model: str,
    tasks: list[str],
    *,
    num_fewshot: int | None = None,
    batch_size: str | None = "auto",
    device: str | None = None,
    limit: float | None = None,
) -> None:
    """Run LM Evaluation Harness."""
    if lm_eval is None:
        logger.error("lm-eval not installed")
        print("Error: lm-eval not installed. Install with `pip install .[eval]`", file=sys.stderr)
        sys.exit(1)

    logger.info(f"Running lm-eval for model: {model} on tasks: {tasks}")
    print(f"Running lm-eval for model: {model} on tasks: {tasks}")

    # Map helmholtz model string to lm_eval arguments if needed
    # For now, we assume the user passes a string compatible with lm_eval's --model_args
    # or we might need to adapt "provider:model" to what lm_eval expects.
    # lm_eval usually expects "model_type", "model_args".

    # Simple pass-through for now, assuming user knows lm_eval model strings
    # or we construct them.

    # Example: if model is "openai:gpt-4o", lm_eval might expect:
    # model="openai-chat-completions", model_args="model=gpt-4o"

    lm_model = model
    model_args = ""

    if ":" in model:
        provider, model_name = model.split(":", 1)
        if provider in ["openai", "anthropic", "google", "ollama", "blablador"]:
            lm_model = "openai-chat-completions"
            model_args = f"model={model_name}"

    results = simple_evaluate(
        model=lm_model,
        model_args=model_args,
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        device=device,
        limit=limit,
    )
    logger.info("LM Evaluation completed")

    # Print results table
    if results:
        from lm_eval.utils import make_table

        print(make_table(results))
