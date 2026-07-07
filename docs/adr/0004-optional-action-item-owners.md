# Optional action-item owners; tasks are the contract

`extraction.json` keeps `owner` in the schema for PRD compatibility, but v1 does not
optimize for filling it. Post-recording Whisper has no speaker labels; Teams active-speaker
highlight is not used for attribution (see ADR 0005).

**Decision:** Ship actionable `task` text (and `deadline` when spoken). Leave `owner` empty
unless explicitly assigned in speech. The team assigns owners in daily standup.

**Consequences:** Published briefs and Cursor context are task lists, not RACI. Do not eval
or prompt-hack for owner fill rate. Avoid inventing owners — increases hallucination risk
for little gain in our process.
