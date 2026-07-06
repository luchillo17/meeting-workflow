# Meeting Workflow

Meeting Recordings (any source) to structured, AI-usable Extractions on local hardware.

## Language

**Meeting Recording**: Video file from any source (typically `.mp4`, `.mkv`) that documents a meeting.
_Avoid_: File, video, input

**Workflow Run**: One end-to-end run on one Meeting Recording.
_Avoid_: Pipeline run, processing run, job

**Extraction**: Full output bundle for one Workflow Run under `output/<slug>/`.
_Avoid_: Output, result set, artifacts folder

**Transcript**: Speech-to-text of Meeting Recording.
_Avoid_: Transcription (verb)

**Visual Capture**: Selected frames + vision-model descriptions of on-screen content.
_Avoid_: Screenshots, frames (alone)

**Structured Extraction**: Machine-readable `extraction.json` — schema-faithful, AI-first.
_Avoid_: JSON output, metadata

**Summary**: Human-readable `summary.md` — same facts as Structured Extraction.
_Avoid_: Report, notes

## Development

### Quality checks: git hooks over CI

Use **local git hooks** ([pre-commit](https://pre-commit.com/)), not **GitHub Actions**, for lint, format, dependency hygiene. Free CI minutes limited; active repos pay. Hooks run on dev machine at commit.

- **Hooks:** ruff, deptry, formatting, other fast static checks
- **CI only when hooks fail:** multi-OS matrix, deploy, cloud-only checks
- **Manual:** GPU pilots, full Workflow Runs on real recordings, heavy analyzers (`pyscn`) — not gated in CI

Install: `uv run pre-commit install`. All hooks: `uv run pre-commit run --all-files`.
