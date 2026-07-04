"""Render Structured Extraction as human-readable Summary markdown."""

from __future__ import annotations


def render_summary(extraction: dict) -> str:
    lines = [f"# {extraction.get('topic') or 'Meeting Summary'}", ""]
    if extraction.get("meeting_date"):
        lines.extend([f"**Date:** {extraction['meeting_date']}", ""])

    sections = [
        ("Key Decisions", extraction.get("key_decisions", [])),
        ("Action Items", _format_action_items(extraction.get("action_items", []))),
        ("Blockers & Risks", extraction.get("blockers_risks", [])),
        ("Status Updates", extraction.get("status_updates", [])),
        ("Technical Details", extraction.get("technical_details", [])),
        ("Open Questions", extraction.get("open_questions", [])),
        ("Next Steps", extraction.get("next_steps", [])),
    ]
    for title, items in sections:
        if items:
            lines.append(f"## {title}")
            lines.append("")
            for item in items:
                lines.append(f"- {item}")
            lines.append("")

    visual = extraction.get("visual_content", [])
    if visual:
        lines.append("## Visual Content")
        lines.append("")
        for entry in visual:
            ts = entry.get("timestamp", "")
            desc = entry.get("description", "")
            lines.append(f"- [{ts}] {desc}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _format_action_items(items: list) -> list[str]:
    formatted = []
    for item in items:
        if isinstance(item, dict):
            owner = item.get("owner", "")
            task = item.get("task", "")
            deadline = item.get("deadline", "")
            parts = [p for p in (owner, task, deadline) if p]
            formatted.append(" — ".join(parts) if parts else str(item))
        else:
            formatted.append(str(item))
    return formatted
