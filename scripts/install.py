#!/usr/bin/env python3
"""Cross-platform first-time install helper."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def _venv_python(root: Path) -> Path:
    name = "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
    return root / ".venv" / name


def _stale_venv_help() -> str:
    if sys.platform == "win32":
        return (
            "Fully quit Cursor, then in an external terminal:\n"
            "  Remove-Item -Recurse -Force .venv\n"
            "  uv sync"
        )
    return "Remove .venv and run: uv sync"


def stale_venv_message(root: Path) -> str | None:
    """Return fix instructions when .venv exists but its Python launcher is broken."""
    py = _venv_python(root)
    if not py.is_file():
        return None
    try:
        result = subprocess.run(
            [str(py), "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=root,
        )
    except OSError:
        return _stale_venv_help()
    output = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0 or "Unable to create process" in output:
        return _stale_venv_help()
    return None


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    uv = shutil.which("uv")
    if uv:
        print("Running uv sync...")
        result = subprocess.run([uv, "sync"], cwd=root, check=False)
        if result.returncode != 0:
            return result.returncode
        stale = stale_venv_message(root)
        if stale:
            print("Stale virtualenv detected (common after renaming the project folder).")
            print("Delete .venv and recreate — do not rename it to .venv.broken.")
            print(stale)
            return 1
        if (root / ".git").is_dir():
            print("Installing pre-commit git hook...")
            hook = subprocess.run(
                [uv, "run", "pre-commit", "install"],
                cwd=root,
                check=False,
            )
            if hook.returncode != 0:
                print("Warning: pre-commit install failed — run: uv run pre-commit install")
        runner = [uv, "run", "meeting-workflow", "setup", *sys.argv[1:]]
    else:
        print("uv not found — install from https://docs.astral.sh/uv/")
        print("Then run: uv sync && uv run meeting-workflow setup")
        runner = [sys.executable, "-m", "workflow.cli", "setup", *sys.argv[1:]]

    print("Running setup...")
    return subprocess.run(runner, cwd=root, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
