# Batch run statistics for performance and quality tracking

Full-folder batches (20+ recordings) lack longitudinal timing and quality data.
Manual ETA from a single cold-start step is unreliable; recording lengths and stage
costs vary widely. We cannot compare runs after `config.yaml` changes (models,
`max_speakers`, frame caps, map-reduce thresholds) or spot regressions over time.

**Decision:** Persist a structured **run record** for each `process` / batch
invocation — performance timings and output-quality signals — in the local
filesystem next to existing artifacts (ADR 0003). No manifest database in v1.

## Performance metrics

**Run-level**

- `run_id`, `started_at`, `finished_at`, `status` (`completed` | `interrupted` | `failed`)
- Input: folder path, recording count, `force` / `extract_only` / `vision_only`
- Config fingerprint: whisper model, diarization on/off, Ollama models, batch workers
- Aggregates: total wall time, steps completed, optional `batch-run.log` path

**Per recording**

- Source: filename, audio duration (from `audio.wav` or segment end), file size
- Wall time per stage: transcript (incl. diarization), frames, vision, extraction
- Counts: transcript segments, diarization turns / distinct speakers, frames kept,
  vision frames analyzed, extraction mode (`single` | `map_reduce`)

## Quality metrics

Complements pilot eval guards (ADR 0006) — longitudinal signals, not a substitute
for golden-file or LLM-as-judge scoring.

**Per recording**

- Pilot eval guard result when applicable: pass/fail + failure reasons
- `grounding_ratio` when extraction grounding is enabled (ADR 0008)
- Extraction shape: chapters, decisions, action items, blockers (counts only)
- Optional flags: diarization skipped, vision tiles skipped, interrupted mid-stage

## Storage

```
output/
  batch-runs/
    <run_id>.json        # run summary + per-recording rows
  <slug>/
    …                    # existing per-meeting artifacts unchanged
```

Append-only `batch-runs.jsonl` is acceptable for a simple history. Keep records
human-readable JSON for shell and Cursor inspection.

## How we use it

- **ETA:** rolling median of stage seconds per audio minute (and per stage), not
  `elapsed ÷ completed_steps` from a cold start. Live progress may consume this
  once history exists ([issue 08](../issues/08-batch-progress-multiline.md)).
- **Tuning:** compare runs after config changes; spot diarization over-clustering
  vs attendee count (ADR 0009).
- **Quality:** track eval pass rate and grounding ratios over time; spot-check
  individual meetings manually when guards pass but bullets look wrong.

## Not in scope (v1)

- Cloud metrics or dashboards
- Automatic quality scoring beyond eval guards / grounding
- Cross-machine aggregation (local-first only, ADR 0001)

**Consequences:** Small JSON write per run; runner hooks record timings as stages
complete. Query recent runs with `meeting-workflow stats`. **Status:** implemented
in `workflow/batch_stats.py`.
