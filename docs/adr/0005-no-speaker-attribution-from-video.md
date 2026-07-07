# No speaker attribution from video tiles in v1

Owner/deadline attribution from "who was highlighted in Teams" would need dense frame
sampling, active-speaker detection, and OCR on roster tiles — brittle across layouts
(gallery vs content-only, screen share hides roster). Our pipeline deliberately works
against that signal:

- Vision prompt skips participant-only tiles; extraction filters tile-only descriptions.
- Frame dedup blurs active-speaker highlight rings (`dedup_layout_blur`, `dedup_global`).
- Transcript segments are `{start, end, text}` only — no diarization.

**Decision:** v1 is post-recording `.mp4` only. No meeting bot, no pyannote diarization,
no video-based speaker-to-name mapping.

**Future (if needed):** calendar roster hints → audio diarization → live bot/API speaking
events — in that order, not Teams highlight chasing first.
