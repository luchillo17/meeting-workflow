# Deterministic token grounding after LLM extraction

LLM grounding pass (ADR 0007) is soft. After merge/grounding/retry, apply a token-overlap
filter: drop bullets whose content tokens are not found in chapter transcript text
(technical_details may also match vision descriptions).

**Decision:** `deterministic_grounding: true` by default (`grounding_min_overlap: 0.34`).
Eval may require `min_grounding_ratio` on pilot folders. Complements — does not replace —
LLM grounding.

**Consequences:** Paraphrased items with low token overlap may be dropped; tune overlap or
disable per run via `config.yaml`. Re-run `--extract-only` after changes.
