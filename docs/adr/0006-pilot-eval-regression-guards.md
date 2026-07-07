# Pilot eval as regression guards, not accuracy scoring

`meeting-workflow eval` checks four pilot output folders: minimum counts, chapter/vision
presence, domain terms, forbidden SaaS artifacts, action-item shape. It is a **floor**
for regressions when prompts, models, or filters change — not golden-file accuracy or
LLM-as-judge quality.

**Decision:** Gate `inbox --eval` on these guards. Spot-check new meetings manually for
task wording and grounding. Do not block backlog on owner fill or semantic dedup perfection.

**Consequences:** Eval can pass while individual bullets are wrong; tightening requires
golden snapshots or `min_grounding_ratio` (ADR 0008), not more count thresholds alone.
