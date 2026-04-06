"""
System context text renderer (Phase 2A).

Converts a SystemContextBundle into a compact, deterministic text block
safe for LLM prompt injection.  No raw JSON — every section is rendered
as clean, human-readable text.

Design:
  - Deterministic (no LLM call, no randomness)
  - Safe (strips HTML, normalizes whitespace, truncates)
  - Short (capped at MAX_BLOCK_CHARS)
  - Does NOT log raw content
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.schemas.system_context import SystemContextBundle

# ── Limits ────────────────────────────────────────────────────────────

MAX_BLOCK_CHARS = 2000
MAX_ATTRIBUTES = 5
MAX_SCOPES = 10
MAX_PERMISSIONS = 8
MAX_METRICS = 6
MAX_RECORDS = 4
MAX_WORKFLOWS = 4

# ── HTML / whitespace cleanup ─────────────────────────────────────────

_RE_HTML = re.compile(r"<[^>]+>")
_RE_MULTI_WS = re.compile(r"\s{2,}")


def _clean(text: str | None) -> str:
    if not text:
        return ""
    text = _RE_HTML.sub("", text)
    text = _RE_MULTI_WS.sub(" ", text)
    return text.strip()


def _trunc(text: str, limit: int = 200) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


# ── Public API ────────────────────────────────────────────────────────


def render_system_context_block(bundle: SystemContextBundle | None) -> str:
    """Render a bundle into a text block for prompt injection.

    Returns empty string if bundle is None or has no useful content.
    """
    if bundle is None:
        return ""

    sections: list[str] = []

    # ── User ──────────────────────────────────────────────────────
    if bundle.user is not None:
        u = bundle.user
        parts: list[str] = []
        if u.display_name:
            parts.append(f"Name: {_clean(u.display_name)}")
        if u.role:
            parts.append(f"Role: {_clean(u.role)}")
        if u.roles:
            parts.append(f"Roles: {', '.join(u.roles[:5])}")
        if u.scopes:
            scopes_text = ", ".join(u.scopes[:MAX_SCOPES])
            parts.append(f"Scopes: {scopes_text}")
        if u.attributes:
            attr_lines = []
            for k, v in list(u.attributes.items())[:MAX_ATTRIBUTES]:
                attr_lines.append(f"  {_clean(str(k))}: {_trunc(_clean(str(v)), 80)}")
            if attr_lines:
                parts.append("Attributes:\n" + "\n".join(attr_lines))
        if parts:
            sections.append("## Current User\n" + "\n".join(parts))

    # ── Tenant ────────────────────────────────────────────────────
    if bundle.tenant is not None:
        t = bundle.tenant
        parts = []
        if t.tenant_name:
            parts.append(f"Organization: {_clean(t.tenant_name)}")
        if t.tenant_slug:
            parts.append(f"Slug: {_clean(t.tenant_slug)}")
        if parts:
            sections.append("## Tenant\n" + "\n".join(parts))

    # ── Permissions (summary only) ────────────────────────────────
    if bundle.has_permissions and bundle.permissions:
        decisions = bundle.permissions.decisions[:MAX_PERMISSIONS]
        if decisions:
            perm_lines = []
            for d in decisions:
                status = "allowed" if d.allowed else "denied"
                perm_lines.append(f"- {d.resource_type}/{d.action}: {status}")
            sections.append("## Permissions\n" + "\n".join(perm_lines))

    # ── Stats ─────────────────────────────────────────────────────
    if bundle.has_stats and bundle.tenant_stats:
        metrics = bundle.tenant_stats.metrics[:MAX_METRICS]
        if metrics:
            stat_lines = []
            for m in metrics:
                label = m.label or m.key
                unit = f" {m.unit}" if m.unit else ""
                stat_lines.append(f"- {_clean(label)}: {m.value}{unit}")
            sections.append("## System Stats\n" + "\n".join(stat_lines))

    # ── Records ───────────────────────────────────────────────────
    if bundle.records:
        record_lines = []
        for r in bundle.records[:MAX_RECORDS]:
            title = _clean(r.title) or r.record_id
            status = f" [{_clean(r.status)}]" if r.status else ""
            summary = f" — {_trunc(_clean(r.summary or ''), 100)}" if r.summary else ""
            record_lines.append(f"- {r.record_type} {title}{status}{summary}")
        if record_lines:
            sections.append("## Recent Records\n" + "\n".join(record_lines))

    # ── Workflows ─────────────────────────────────────────────────
    if bundle.workflows:
        wf_lines = []
        for w in bundle.workflows[:MAX_WORKFLOWS]:
            parts_w = [f"- {_clean(w.workflow_type)}"]
            if w.total is not None:
                parts_w.append(f"total={w.total}")
            if w.pending_count is not None:
                parts_w.append(f"pending={w.pending_count}")
            if w.completed_count is not None:
                parts_w.append(f"completed={w.completed_count}")
            wf_lines.append(" ".join(parts_w))
        sections.append("## Workflows\n" + "\n".join(wf_lines))

    if not sections:
        return ""

    block = "\n\n".join(sections)

    # Final safety truncation
    if len(block) > MAX_BLOCK_CHARS:
        block = block[:MAX_BLOCK_CHARS] + "\n…(truncated)"

    return block
