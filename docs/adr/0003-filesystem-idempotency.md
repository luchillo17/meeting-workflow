# Filesystem idempotency without a manifest database

A Workflow Run skips processing when `extraction.json` already exists in the target Extraction directory. `--force` reprocesses. The filesystem is the source of truth in v1 — no SQLite manifest, no cross-folder hash dedup. Folder scanning and a manifest may be added post-pilot if volume warrants it.
