# Future: meeting knowledge to Cursor and second brain

Post-v1 direction, 2026-07. Not scheduled — architecture for future **publish** + downstream consumption.

## Goal

Tasks + decisions from meetings into **AI coding agents** (Cursor) reliably. No per-seat SaaS lock-in.

## Stack

```text
Meeting recording .mp4 (or optional live bot)
  → meeting-workflow (transcript + visual capture + extraction.json)
  → publish → app-repo/docs/meetings/<date>-<slug>.md (+ optional .json)
  → Cursor @docs/meetings/... for implementation tasks
  → optional second brain (Obsidian / Notion) for human linking & tasks
```

## vs SaaS-only (Read, Fireflies, Fathom)

| Need                                              | Pipeline + publish      | SaaS → Notion only   |
| ------------------------------------------------- | ----------------------- | -------------------- |
| Structured `action_items` (owner, task, deadline) | Yes (`extraction.json`) | Generic shape        |
| Vision / whiteboard in extraction                 | Yes                     | Speech-first         |
| Cursor `@` / repo index                           | Yes, in product repo    | Export glue required |
| Privacy / local capture                           | Yes                     | Cloud                |
| Fidelity (Spanish, no bot-as-attendee)            | Configurable prompts    | Opaque               |

SaaS = capture or team UI. Repo publish = coding agents.

## Three tiers

| Tier        | Content                                            | App repo?            | Use                    |
| ----------- | -------------------------------------------------- | -------------------- | ---------------------- |
| Agent brief | Published markdown from `summary.md` + frontmatter | Yes                  | Default Cursor context |
| Structured  | `extraction.json` sidecar                          | Optional             | Scripts, validation    |
| Archive     | Full `transcript.json`, frames                     | No (local `output/`) | Debug, RAG later       |

Do not commit full transcripts to app repo by default — noisy for retrieval.

## Second brain (Notion / Obsidian)

- Not for Cursor — agents index repo, not Notion
- Humans: link meetings to projects, MOCs, task DBs
- Pattern: one extraction, two publishes (brain + repo); source of truth `extraction.json`

## RAG

- Not needed for single-meeting coding (`@` one published file)
- Add at 50+ meetings or cross-meeting transcript recall
- Structured extraction = pre-RAG compression; chunk transcripts if indexing

## Cloud LLM for text-only extraction

Optional hybrid: local Whisper + vision; cloud pass for extraction (~$0.02–0.10/meeting). Same `extraction.json` shape.

## Likely next

1. `meeting-workflow publish` — copy/enrich `extraction.json` + `summary.md` to `docs/meetings/`; update `index.md`
2. Optional adapters: Obsidian vault, Notion API
3. Optional `index` for transcript RAG / MCP search

## References

- Output: [README.md](../../README.md)
- Schema: [workflow/extraction.py](../../workflow/extraction.py)
- PRD out of scope v1: [PRD.md](../PRD.md)
