import subprocess  # nosec B404
import sys


def start_proxy(
    model: str, port: int = 4000, config_path: str | None = None, debug: bool = False
) -> None:
    """Start LiteLLM Proxy."""
    cmd = ["litellm", "--model", model, "--port", str(port)]

    if config_path:
        cmd.extend(["--config", config_path])

    if debug:
        cmd.append("--debug")

    print(f"Starting LiteLLM proxy for {model} on port {port}...")
    try:
        subprocess.run(cmd, check=True)  # nosec B603
    except FileNotFoundError:
        print("Error: litellm not installed. Install with `pip install .[proxy]`", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nStopping proxy...")
