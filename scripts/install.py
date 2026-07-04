#!/usr/bin/env python3
"""Cross-platform first-time install helper."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    uv = shutil.which("uv")
    if uv:
        print("Running uv sync...")
        result = subprocess.run([uv, "sync"], cwd=root, check=False)
        if result.returncode != 0:
            return result.returncode
        runner = [uv, "run", "meeting-workflow", "setup", *sys.argv[1:]]
    else:
        print("uv not found — install from https://docs.astral.sh/uv/")
        print("Then run: uv sync && uv run meeting-workflow setup")
        runner = [sys.executable, "-m", "workflow.cli", "setup", *sys.argv[1:]]

    print("Running setup...")
    return subprocess.run(runner, cwd=root, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
