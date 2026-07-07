"""Cross-platform CLI for Meeting Workflow."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from rich.console import Console

from workflow.cuda_paths import ensure_cuda_dll_paths
from workflow.extraction import OllamaStructuredExtractor
from workflow.extraction_eval import evaluate_pilot_outputs
from workflow.frames import FfmpegFrameExtractor, frames_bundle_valid, load_transcript_for_frames
from workflow.preflight import Check, has_failures, run_preflight
from workflow.publish import discover_publishable, publish_output_dir
from workflow.runner import WorkflowRunner
from workflow.scan import (
    discover_recordings,
    pending_recordings,
    resolve_watch_folders,
    scan_recordings,
)
from workflow.settings import Settings
from workflow.transcriber import WhisperTranscriber
from workflow.utils import invalidate_downstream_artifacts, slugify
from workflow.vision import OllamaVisionAnalyzer

_STATUS_STYLE = {"ok": "green", "warn": "yellow", "fail": "red"}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="meeting-workflow",
        description="Meeting Workflow — process meeting recordings locally",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    process = sub.add_parser("process", help="Run workflow on one or more Meeting Recordings")
    process.add_argument(
        "--file",
        type=Path,
        action="append",
        dest="files",
        metavar="PATH",
        help="Path to .mp4 (repeat for batch; overrides RECORDING_PATH)",
    )
    process.add_argument("--force", action="store_true", help="Reprocess even if Extraction exists")
    process.add_argument(
        "--extract-only",
        action="store_true",
        help="Re-run extraction only (reuse transcript, frames, and vision)",
    )
    process.add_argument(
        "--folder",
        type=Path,
        action="append",
        dest="folders",
        metavar="PATH",
        help="Process pending recordings from folder (WATCH_FOLDERS / ONEDRIVE_ROOT fallback)",
    )

    scan_cmd = sub.add_parser(
        "scan", help="List recordings in watch folders and their processing status"
    )
    scan_cmd.add_argument(
        "--folder",
        type=Path,
        action="append",
        dest="folders",
        metavar="PATH",
        help="Folder to scan (repeatable; default: WATCH_FOLDERS or ONEDRIVE_ROOT/Grabaciones)",
    )
    scan_cmd.add_argument(
        "--recursive",
        action="store_true",
        help="Include recordings in subfolders",
    )
    scan_cmd.add_argument(
        "--process",
        action="store_true",
        help="Run workflow on pending recordings found by scan",
    )

    frames = sub.add_parser(
        "frames", help="Extract frames only (uses existing transcript.json if present)"
    )
    frames.add_argument("--file", type=Path, help="Path to .mp4 (overrides RECORDING_PATH)")
    frames.add_argument(
        "--output-dir",
        type=Path,
        help="Extraction output directory (default: OUTPUT_DIR/<slug>/)",
    )
    frames.add_argument(
        "--force", action="store_true", help="Re-extract even if frames.json exists"
    )

    sub.add_parser("check", help="Verify prerequisites and configuration")

    setup = sub.add_parser("setup", help="First-time setup (.env, output dir, optional model pull)")
    setup.add_argument(
        "--pull-models",
        action="store_true",
        help="Run ollama pull for configured text and vision models",
    )

    eval_cmd = sub.add_parser(
        "eval", help="Run pilot extraction spot-checks against output folders"
    )
    eval_cmd.add_argument(
        "--output-dir",
        type=Path,
        help="Output root to scan (default: OUTPUT_DIR from settings)",
    )

    publish_cmd = sub.add_parser(
        "publish", help="Publish extraction briefs to docs/meetings for Cursor context"
    )
    publish_cmd.add_argument(
        "--dir",
        type=Path,
        action="append",
        dest="dirs",
        metavar="PATH",
        help="Extraction output folder (repeatable; default: all under OUTPUT_DIR)",
    )
    publish_cmd.add_argument(
        "--meetings-dir",
        type=Path,
        default=Path("docs/meetings"),
        help="Destination for published briefs (default: docs/meetings)",
    )
    publish_cmd.add_argument(
        "--json",
        action="store_true",
        help="Also write extraction.json sidecar next to the markdown brief",
    )
    publish_cmd.add_argument(
        "--all",
        action="store_true",
        help="Publish every folder under OUTPUT_DIR that has extraction.json",
    )

    return parser


def _print_checks(console: Console, checks: list[Check]) -> None:
    for check in checks:
        label = check.status.upper()
        console.print(f"[{_STATUS_STYLE[check.status]}]{label:4}[/] {check.name}: {check.detail}")


def cmd_check(_args: argparse.Namespace) -> int:
    console = Console()
    checks = run_preflight()
    _print_checks(console, checks)
    if has_failures(checks):
        console.print("[red]Fix failed checks before running process.[/red]")
        return 1
    console.print("[green]Ready to run process.[/green]")
    return 0


def cmd_setup(args: argparse.Namespace) -> int:
    console = Console()
    root = Path.cwd()
    env_example = root / ".env.example"
    env_path = root / ".env"

    if not env_path.exists() and env_example.exists():
        env_path.write_text(env_example.read_text(encoding="utf-8"), encoding="utf-8")
        console.print(f"[green]Created[/green] {env_path}")
    elif env_path.exists():
        console.print(f"[yellow]Exists[/yellow] {env_path}")
    else:
        console.print("[red]Missing .env.example — cannot create .env[/red]")
        return 1

    settings = Settings.load(root)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[green]Output dir[/green] -> {settings.output_dir.resolve()}")

    if args.pull_models:
        ollama = shutil.which("ollama")
        if not ollama:
            console.print("[red]ollama not found in PATH[/red]")
            return 1
        seen: set[str] = set()
        for model in (settings.ollama_text_model, settings.ollama_vision_model):
            if model in seen:
                continue
            seen.add(model)
            console.print(f"[cyan]Pulling[/cyan] {model}")
            result = subprocess.run([ollama, "pull", model], check=False)
            if result.returncode != 0:
                console.print(f"[red]Failed to pull {model}[/red]")
                return 1

    console.print("[green]Setup complete.[/green] Edit .env, then run: meeting-workflow check")
    return 0


def _config_with_env(settings: Settings) -> dict:
    merged = dict(settings.config)
    ollama = dict(merged.get("ollama", {}))
    ollama["base_url"] = settings.ollama_base_url
    ollama["text_model"] = settings.ollama_text_model
    ollama["vision_model"] = settings.ollama_vision_model
    merged["ollama"] = ollama
    return merged


def _resolve_watch_folders(cli_folders: list[Path] | None) -> list[Path]:
    return resolve_watch_folders(
        cli_folders=cli_folders,
        watch_folders_env=os.environ.get("WATCH_FOLDERS"),
        onedrive_root=os.environ.get("ONEDRIVE_ROOT"),
    )


def _run_workflow_batch(
    settings: Settings,
    files: list[Path],
    *,
    force: bool,
    extract_only: bool,
) -> int:
    config = _config_with_env(settings)
    transcriber = WhisperTranscriber(config)
    runner = WorkflowRunner(
        settings,
        transcriber,
        FfmpegFrameExtractor(config),
        OllamaVisionAnalyzer(config),
        OllamaStructuredExtractor(config),
    )
    try:
        runner.run_batch(files, force=force, extract_only=extract_only)
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_process(args: argparse.Namespace) -> int:
    ensure_cuda_dll_paths()
    settings = Settings.load()
    files: list[Path] = list(args.files or [])
    if args.folders:
        folders = _resolve_watch_folders(args.folders)
        if not folders:
            print(
                "Error: provide --folder or set WATCH_FOLDERS / ONEDRIVE_ROOT in .env",
                file=sys.stderr,
            )
            return 1
        if args.force:
            files = discover_recordings(folders)
        else:
            files = pending_recordings(folders, settings.output_dir)
    elif not files:
        recording = Path(os.environ.get("RECORDING_PATH", ""))
        if recording and str(recording) != ".":
            files = [recording]
    if not files:
        print(
            "Error: provide --file, --folder, or set RECORDING_PATH in .env",
            file=sys.stderr,
        )
        return 1

    return _run_workflow_batch(settings, files, force=args.force, extract_only=args.extract_only)


def cmd_scan(args: argparse.Namespace) -> int:
    console = Console()
    settings = Settings.load()
    folders = _resolve_watch_folders(args.folders)
    if not folders:
        print(
            "Error: provide --folder or set WATCH_FOLDERS / ONEDRIVE_ROOT in .env",
            file=sys.stderr,
        )
        return 1

    results = scan_recordings(folders, settings.output_dir, recursive=args.recursive)
    if not results:
        console.print("[yellow]No recordings found[/yellow] in configured watch folders")
        return 0

    for result in results:
        style = "green" if result.status == "done" else "yellow"
        console.print(
            f"[{style}]{result.status:7}[/{style}] {result.recording.name} "
            f"[dim]-> {result.output_dir.name}[/dim]"
        )

    pending = [result.recording for result in results if result.status == "pending"]
    console.print(
        f"[bold]{len(results)}[/bold] recording(s): "
        f"[green]{len(results) - len(pending)} done[/green], "
        f"[yellow]{len(pending)} pending[/yellow]"
    )

    if args.process:
        if not pending:
            console.print("[green]Nothing pending — all recordings already processed.[/green]")
            return 0
        ensure_cuda_dll_paths()
        return _run_workflow_batch(settings, pending, force=False, extract_only=False)
    return 0


def cmd_frames(args: argparse.Namespace) -> int:
    console = Console()
    settings = Settings.load()
    recording = args.file or Path(os.environ.get("RECORDING_PATH", ""))
    if not recording or str(recording) == ".":
        print("Error: provide --file or set RECORDING_PATH in .env", file=sys.stderr)
        return 1
    if not recording.is_file():
        print(f"Error: Meeting Recording not found: {recording}", file=sys.stderr)
        return 1

    output_dir = args.output_dir or (settings.output_dir / slugify(recording.name))
    if frames_bundle_valid(output_dir) and not args.force:
        console.print(
            f"[yellow]Skipping[/yellow] - frames already exist at {output_dir / 'frames.json'}"
        )
        console.print("Use --force to re-extract.")
        return 0

    invalidate_downstream_artifacts(output_dir)

    transcript = load_transcript_for_frames(output_dir)
    if transcript.segments:
        count = len(transcript.segments)
        console.print(f"[dim]Using transcript.json for visual-cue boost ({count} segments)[/dim]")
    else:
        console.print("[dim]No transcript.json — scene + interval sampling only[/dim]")

    config = _config_with_env(settings)
    extractor = FfmpegFrameExtractor(config)
    try:
        paths = extractor.extract(recording, transcript, output_dir)
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    console.print(f"[green]Extracted {len(paths)} frame(s)[/green] -> {output_dir / 'frames'}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    console = Console()
    settings = Settings.load()
    output_root = args.output_dir or settings.output_dir
    failures = evaluate_pilot_outputs(output_root)
    if not failures:
        console.print(f"[green]Pilot eval passed[/green] for {output_root}")
        return 0
    console.print(f"[red]Pilot eval failed[/red] for {output_root}")
    for failure in failures:
        console.print(f"  - {failure}")
    return 1


def cmd_publish(args: argparse.Namespace) -> int:
    console = Console()
    settings = Settings.load()
    meetings_dir = args.meetings_dir
    if args.all or not args.dirs:
        targets = discover_publishable(settings.output_dir)
    else:
        targets = list(args.dirs)

    if not targets:
        print(f"Error: no publishable folders under {settings.output_dir}", file=sys.stderr)
        return 1

    try:
        for output_dir in targets:
            markdown_path = publish_output_dir(output_dir, meetings_dir, include_json=args.json)
            console.print(f"[green]Published[/green] {markdown_path}")
        console.print(f"[green]Updated[/green] {meetings_dir / 'index.md'}")
    except (FileNotFoundError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "check":
        return cmd_check(args)
    if args.command == "setup":
        return cmd_setup(args)
    if args.command == "process":
        return cmd_process(args)
    if args.command == "frames":
        return cmd_frames(args)
    if args.command == "eval":
        return cmd_eval(args)
    if args.command == "publish":
        return cmd_publish(args)
    if args.command == "scan":
        return cmd_scan(args)
    return 1
