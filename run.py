"""CLI entrypoint for Meeting Workflow."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from workflow.fixtures import fixture_adapters
from workflow.runner import WorkflowRunner
from workflow.settings import Settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Meeting Workflow — process Teams Meeting Recordings")
    sub = parser.add_subparsers(dest="command", required=True)

    process = sub.add_parser("process", help="Run workflow on one Meeting Recording")
    process.add_argument("--file", type=Path, help="Path to .mp4 (overrides RECORDING_PATH)")
    process.add_argument("--force", action="store_true", help="Reprocess even if Extraction exists")

    args = parser.parse_args(argv)

    if args.command == "process":
        settings = Settings.load()
        recording = args.file or Path(os.environ.get("RECORDING_PATH", ""))
        if not recording or str(recording) == ".":
            print("Error: provide --file or set RECORDING_PATH in .env", file=sys.stderr)
            return 1

        transcriber, frames, vision, extractor = fixture_adapters()
        runner = WorkflowRunner(settings, transcriber, frames, vision, extractor)
        try:
            runner.run(recording, force=args.force)
        except FileNotFoundError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
