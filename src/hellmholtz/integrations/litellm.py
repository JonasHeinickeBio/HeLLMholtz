"""LiteLLM proxy integration.

Runs a local LiteLLM proxy in front of any model so clients can reach it
through an OpenAI-compatible (``/v1/chat/completions``) or an
Anthropic-compatible (``/v1/messages``, e.g. Claude Code via
``ANTHROPIC_BASE_URL``) endpoint.
"""

import logging
import os
from pathlib import Path
import re
import secrets
import subprocess  # nosec B404
import sys
import tempfile

logger = logging.getLogger(__name__)

MASTER_KEY_ENV_VAR = "LITELLM_MASTER_KEY"
_GENERATED_KEY_PREFIX = "sk-hellm-"
# Unique slot in the command list, replaced with the temp config path.
_CONFIG_PATH_PLACEHOLDER = "__hellmholtz-proxy-config__"
# Newlines and other control characters would break the generated YAML config.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f]")


def generate_master_key() -> str:
    """Generate a random proxy master key.

    Returns:
        A key of the form ``sk-hellm-<random>`` suitable as proxy ``master_key``.
    """
    return f"{_GENERATED_KEY_PREFIX}{secrets.token_urlsafe(24)}"


def _validate_proxy_value(value: str, label: str) -> None:
    """Validate a value that will be embedded in the generated YAML config.

    Raises:
        ValueError: If the value is empty or contains control characters.
    """
    if not value:
        raise ValueError(f"{label} must not be empty")
    if _CONTROL_CHARS_RE.search(value):
        raise ValueError(f"{label} must not contain newlines or control characters")


def _to_config_model(model: str) -> str:
    """Convert ``provider:model`` to the config-compatible ``provider/model``.

    The LiteLLM CLI accepts colon-prefixed providers (``openai:gpt-4o``), but
    its config loader does not recognize them, so the colon form is rewritten
    for generated config files.
    """
    provider, sep, rest = model.partition(":")
    if sep and provider and rest:
        return f"{provider}/{rest}"
    return model


def build_proxy_config(model: str, model_name: str, master_key: str | None = None) -> str:
    """Build a LiteLLM proxy YAML config exposing ``model`` as ``model_name``.

    Args:
        model: Full LiteLLM model identifier (e.g. ``openai:gpt-4o``).
        model_name: Alias clients should use (e.g. ``claude``).
        master_key: Optional proxy authentication key.

    Returns:
        YAML text for the proxy config.

    Raises:
        ValueError: If any value is empty or contains control characters.
    """
    _validate_proxy_value(model, "model")
    _validate_proxy_value(model_name, "model_name")
    if master_key is not None:
        _validate_proxy_value(master_key, "master_key")

    config = (
        f"model_list:\n  - model_name: {model_name}\n    litellm_params:\n"
        f"      model: {_to_config_model(model)}\n"
    )
    if master_key is not None:
        config += f"general_settings:\n  master_key: {master_key}\n"
    return config


def claude_code_snippet(
    host: str,
    port: int,
    model_name: str,
    master_key: str,
    key_from_env: bool = False,
) -> str:
    """Build the shell snippet that points Claude Code at the proxy.

    Args:
        host: Proxy host.
        port: Proxy port.
        model_name: Model alias exposed by the proxy.
        master_key: Proxy master key (ignored when ``key_from_env`` is True).
        key_from_env: Reference the ``LITELLM_MASTER_KEY`` environment
            variable in the snippet instead of a literal key.

    Returns:
        Shell block that configures Claude Code for the proxy.
    """
    token = f"${MASTER_KEY_ENV_VAR}" if key_from_env else master_key
    return (
        "unset ANTHROPIC_API_KEY\n"
        f'export ANTHROPIC_BASE_URL="http://{host}:{port}"\n'
        f'export ANTHROPIC_AUTH_TOKEN="{token}"\n'
        f'export ANTHROPIC_MODEL="{model_name}"\n'
        "claude"
    )


def _resolve_master_key(master_key: str | None, claude_code: bool) -> tuple[str | None, bool]:
    """Resolve the proxy master key.

    Args:
        master_key: Explicit key, if any.
        claude_code: Whether a missing key should be auto-generated.

    Returns:
        Tuple of ``(key, key_from_env)``. The key is taken from the argument,
        the ``LITELLM_MASTER_KEY`` environment variable, or generated in
        ``claude_code`` mode.
    """
    if master_key is None:
        env_key = os.getenv(MASTER_KEY_ENV_VAR)
        if env_key:
            return env_key, True
    if master_key is None and claude_code:
        return generate_master_key(), False
    return master_key, False


def _print_proxy_banner(
    model: str,
    host: str,
    port: int,
    alias: str,
    master_key: str | None,
    key_from_env: bool,
    claude_code: bool,
) -> None:
    """Print the proxy startup banner, including the Claude Code snippet."""
    base_url = f"http://{host}:{port}"
    print(f"Starting LiteLLM proxy for {model} (alias: {alias}) on {base_url}")
    print(f"  OpenAI-compatible:   {base_url}/v1/chat/completions")
    print(f"  Anthropic-compatible: {base_url}/v1/messages")
    if master_key is not None:
        if claude_code:
            print("\nPoint Claude Code at the proxy (run in a second terminal):")
            print(claude_code_snippet(host, port, alias, master_key, key_from_env))
        else:
            key_ref = f"${MASTER_KEY_ENV_VAR}" if key_from_env else "the provided master key"
            print(f"  Auth: master key required (Authorization: Bearer {key_ref})")
    else:
        print("  Auth: none (open proxy) - only use on trusted networks!")


def _run_proxy_command(command: list[str], model: str, port: int) -> None:
    """Run the litellm proxy command, translating failures into clean exits."""
    logger.info(f"Starting LiteLLM proxy for {model} on port {port}...")
    try:
        subprocess.run(command, check=True)  # nosec B603
    except FileNotFoundError:
        logger.error("LiteLLM not installed")
        print(
            "Error: litellm not installed. Install with `pip install .[proxy]`",
            file=sys.stderr,
        )
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Stopping proxy...")
        print("\nStopping proxy...")
    except subprocess.CalledProcessError as e:
        # A negative return code means the proxy was killed by a signal
        # (e.g. SIGTERM/SIGHUP when its session or parent is stopped) -
        # that is a clean stop, not a failure.
        if e.returncode < 0:
            logger.info(f"Proxy stopped (signal {-e.returncode})")
            print(f"\nProxy stopped (signal {-e.returncode}).")
            return
        raise


def start_proxy(
    model: str,
    port: int = 4000,
    config_path: str | None = None,
    debug: bool = False,
    host: str = "127.0.0.1",
    model_name: str | None = None,
    master_key: str | None = None,
    claude_code: bool = False,
) -> None:
    """Start a LiteLLM proxy for ``model``.

    The proxy exposes both OpenAI-compatible and Anthropic-compatible
    endpoints, so it can serve Claude Code via ``ANTHROPIC_BASE_URL``.

    Args:
        model: Full LiteLLM model identifier (e.g. ``openai:gpt-4o``).
        port: Port to listen on.
        config_path: Path to an existing LiteLLM config file. When given,
            ``model_name`` and ``master_key`` are ignored.
        debug: Run the proxy in debug mode.
        host: Host to bind the proxy to.
        model_name: Alias to expose the model under. When set (or when a
            master key applies) a config file is generated automatically.
        master_key: Proxy authentication key. Falls back to the
            ``LITELLM_MASTER_KEY`` environment variable. In ``claude_code``
            mode a random key is generated when neither is provided.
        claude_code: Print the shell snippet that configures Claude Code for
            this proxy and auto-generate a master key when none is available.

    Raises:
        SystemExit: If litellm is not installed.
    """
    master_key, key_from_env = _resolve_master_key(master_key, claude_code)

    alias = model_name or model
    use_generated_config = config_path is None and (
        model_name is not None or master_key is not None
    )

    cmd = ["litellm"]
    if config_path is not None:
        if model_name is not None or master_key is not None:
            print("Warning: config file given; --name and --master-key are ignored.")
        cmd.extend(["--config", config_path])
    elif use_generated_config:
        cmd.extend(["--config", _CONFIG_PATH_PLACEHOLDER])  # replaced below
    else:
        cmd.extend(["--model", model])
    cmd.extend(["--port", str(port), "--host", host])
    if debug:
        cmd.append("--debug")

    _print_proxy_banner(model, host, port, alias, master_key, key_from_env, claude_code)
    # Ensure the banner (including the auth key) is visible before the proxy
    # blocks, even when stdout is redirected (block-buffered).
    sys.stdout.flush()

    if use_generated_config:
        with tempfile.TemporaryDirectory(prefix="hellmholtz-proxy-") as tmpdir:
            config_file = Path(tmpdir) / "hellmholtz-proxy.yaml"
            config_file.write_text(build_proxy_config(model, alias, master_key))
            cmd[cmd.index(_CONFIG_PATH_PLACEHOLDER)] = str(config_file)
            _run_proxy_command(cmd, model, port)
    else:
        _run_proxy_command(cmd, model, port)
