# Future inspiration: meeting knowledge → Cursor & second brain

Post-v1 direction discussed 2026-07. Not scheduled work — captures architecture for when we add **publish** and downstream consumption.

## Goal

Get **tasks and decisions** from meetings into **AI coding agents** (Cursor) reliably, without per-seat SaaS lock-in.

## Recommended stack

```text
Teams .mp4 (or optional live bot)
  → meeting-workflow (transcript + visual capture + extraction.json)
  → publish → app-repo/docs/meetings/<date>-<slug>.md (+ optional .json)
  → Cursor @docs/meetings/... for implementation tasks
  → optional second brain (Obsidian / Notion) for human linking & tasks
```

## Why this wins vs SaaS-only (Read, Fireflies, Fathom)

| Need                                              | Pipeline + publish      | SaaS → Notion only   |
| ------------------------------------------------- | ----------------------- | -------------------- |
| Structured `action_items` (owner, task, deadline) | Yes (`extraction.json`) | Generic shape        |
| Vision / whiteboard text in extraction            | Yes                     | Speech-first         |
| Cursor `@` / repo index                           | Yes, in product repo    | Export glue required |
| Privacy / local capture                           | Yes                     | Cloud                |
| Fidelity rules (Spanish, no bot-as-attendee)      | Configurable prompts    | Opaque               |

SaaS can feed **capture** or team UI; **repo publish** feeds **coding agents**.

## Three tiers (what goes where)

| Tier        | Content                                            | App repo?            | Use                    |
| ----------- | -------------------------------------------------- | -------------------- | ---------------------- |
| Agent brief | Published markdown from `summary.md` + frontmatter | Yes                  | Default Cursor context |
| Structured  | `extraction.json` sidecar                          | Optional             | Scripts, validation    |
| Archive     | Full `transcript.json`, frames                     | No (local `output/`) | Debug, RAG later       |

Do not commit full transcripts to the app repo by default — noisy for retrieval.

## Second brain (Notion / Obsidian)

- **Not for Cursor directly** — agents index the repo, not Notion.
- **For humans:** link meetings to projects, MOCs, task DBs.
- **Pattern:** one extraction → two publishes (brain + repo), same source of truth in `extraction.json`.

## RAG

- **Not needed** for single-meeting coding tasks (`@` one published file).
- **Add later** when corpus is large (50+ meetings) or questions need **transcript-level** recall across meetings.
- Structured extraction is pre-RAG compression; chunk **transcripts** if indexing.

## Cloud LLM for text-only extraction

- Optional hybrid: local Whisper + vision; cloud pass for extraction (~$0.02–0.10/meeting).
- Still publish same `extraction.json` shape.

## Likely next implementation

1. `meeting-workflow publish` — copy/enrich `extraction.json` + `summary.md` to configurable `docs/meetings/` path; update `index.md`.
2. Optional adapters: Obsidian vault, Notion API (second brain).
3. Optional `index` command for transcript RAG / MCP search (scale).

## References

- Output layout: [README.md](../../README.md)
- Structured schema: [workflow/extraction.py](../../workflow/extraction.py)
- PRD out of scope (v1): publish, RAG, NotebookLM — [PRD.md](../PRD.md)
