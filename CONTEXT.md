# Meeting Workflow

Turns Teams Meeting Recordings into structured, AI-usable Extractions on local hardware.

## Language

**Meeting Recording**:
A Teams-sourced video file (typically `.mp4`) that documents a meeting.
_Avoid_: File, video, input

**Workflow Run**:
One end-to-end execution of the workflow on a single Meeting Recording.
_Avoid_: Pipeline run, processing run, job

**Extraction**:
The complete output bundle for one Workflow Run, written under `output/<slug>/`.
_Avoid_: Output, result set, artifacts folder

**Transcript**:
Speech-to-text representation of the Meeting Recording.
_Avoid_: Transcription (verb form)

**Visual Capture**:
Selected video frames plus vision-model descriptions of on-screen content.
_Avoid_: Screenshots, frames (alone)

**Structured Extraction**:
Machine-readable `extraction.json` — schema-faithful, AI-first.
_Avoid_: JSON output, metadata

**Summary**:
Human-readable `summary.md` rendering the same facts as Structured Extraction.
_Avoid_: Report, notes
