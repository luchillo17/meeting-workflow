## Build

Multiline batch progress for narrow terminals. Long Teams recording titles currently
squeeze the Rich live bar — percentage, elapsed time, and the bar get truncated
when crammed into one `task.description` line (`line[:100]`).

## Proposed layout

1. **Line 1 (stats):** `Progress 4% · step 4/84 · stage 1/4 Transcript · working`
2. **Line 2 (recording):** `[4/21]` + full filename, wrapped at `console.width`
3. **Line 3 (bar row):** spinner + bar + `%` + m/n + elapsed — no long title on this row

## Implementation notes

- **TTY:** multiline `task.description` or `Progress.log()` for the recording line;
  avoid duplicating cyan log lines above the live bar.
- **Piped logs / Tee:** two plain text lines (no truncation, no ANSI bar).
- Rich `TextColumn` in our Rich version does not accept `expand=True`; use
  `Progress(expand=True)` or a custom column if needed.
- File: `workflow/batch_progress.py`
- Optional: show ETA from prior runs once [ADR 0010](../adr/0010-batch-run-statistics.md)
  history exists.

## Acceptance

- [ ] Full recording title visible (wrapped, not ellipsized) on 80-col terminal
- [ ] Bar, percentage, step count, and timer never truncated on narrow widths
- [ ] Log/Tee output remains readable without ANSI bar noise
- [ ] Unit tests for wrap helper + non-TTY two-line output

## Blocked by

Current full-batch run — implement after batch completes (no mid-run code changes).
