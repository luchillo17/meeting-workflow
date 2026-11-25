# Filesystem idempotency without manifest database

A manifest database (SQLite, job queue) adds migration and sync complexity
before we know batch volume or multi-machine needs.

**Decision:** The **filesystem is the source of truth** in v1:

- Each recording → `OUTPUT_DIR/<slug>/` with artifacts (`transcript.json`,
  `frames/`, `extraction.json`, …)
- A Workflow Run **skips** when `extraction.json` already exists
- `--force` reprocesses; `--extract-only` / `--vision-only` reuse upstream artifacts
- Downstream invalidation clears stale vision/extraction when frames are re-run

No cross-folder hash dedup. Folder scan + inbox ([issues 17–18](../issues/README.md))
discover pending recordings by missing `extraction.json`.

**Batch run history** is also filesystem-only: append-only
`output/batch-runs.jsonl` + per-run JSON ([ADR 0010](0010-batch-run-statistics.md)),
not a separate metrics DB.

**Consequences**

- Simple to inspect, copy, and publish; idempotency is visible in Explorer
- Renaming a recording changes the slug — treated as a new meeting
- Post-pilot: revisit manifest DB only if scan volume or dedup requirements grow
