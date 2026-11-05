#!/usr/bin/env python
"""Print the comma-separated ``blablador:<name>`` list of chat models to benchmark.

Reads models_status.yaml (written by the model monitor). Skips unavailable models,
non-chat endpoints (embeddings, rerankers, speech, image/video/OCR) and names that
contain a comma, which the ``--models`` list syntax cannot carry (use their alias).

    select_models.py                    # every available chat model
    select_models.py --new-since HEAD~1 # only models that were not available at that git ref
    select_models.py --models a,b       # explicit subset, still filtered
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess  # nosec B404 - only runs `git show` on a repo path
import sys
from typing import Any

import yaml

NON_CHAT = re.compile(
    r"embed|rerank|whisper|audio|flux|image|ocr|video|voice|anonymizer|clef|ada-002|davinci|gpt-3\.5",
    re.IGNORECASE,
)


def chat_models(status: dict[str, Any]) -> set[str]:
    models = (status or {}).get("models", {})
    return {
        name
        for name, info in models.items()
        if isinstance(info, dict)
        and info.get("available")
        and "," not in name
        and not NON_CHAT.search(name)
    }


def load_at_ref(path: Path, ref: str) -> dict[str, Any]:
    try:
        out = subprocess.run(  # noqa: S603  # nosec B603 B607
            ["git", "show", f"{ref}:{path.as_posix()}"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {}
    return yaml.safe_load(out) or {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--status", type=Path, default=Path("models_status.yaml"))
    parser.add_argument(
        "--new-since", default=None, help="git ref; keep only newly available models"
    )
    parser.add_argument("--models", default=None, help="comma-separated subset of model names")
    args = parser.parse_args()

    names = chat_models(yaml.safe_load(args.status.read_text()) or {})
    if args.new_since:
        names -= chat_models(load_at_ref(args.status, args.new_since))
    if args.models:
        wanted = {
            m.strip().removeprefix("blablador:") for m in args.models.split(",") if m.strip()
        }
        names &= wanted
    print(",".join(f"blablador:{n}" for n in sorted(names)))
    print(f"{len(names)} models selected", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
