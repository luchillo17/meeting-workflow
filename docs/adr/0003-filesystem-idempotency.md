# Filesystem idempotency without manifest database

Workflow Run skips when `extraction.json` exists in target Extraction dir. `--force` reprocesses. Filesystem = source of truth v1 — no SQLite manifest, no cross-folder hash dedup. Folder scan + manifest post-pilot if volume warrants.
