"""
Deterministic answer builder from SystemContextBundle (Phase 2B).

Builds structured, human-readable answers from system context data
WITHOUT calling any LLM.  Used as fallback when:
  - LLM is disabled or fails
  - Question is SYSTEM / ACCESS / MIXED and system context is available
  - No document snippets exist but system context has data

Design:
  - Deterministic — no randomness, no LLM calls
  - Safe — strips HTML, normalizes whitespace, truncates
  - No raw JSON / dict dumps in output
  - Concise — short factual summaries
  - Domain-neutral — no business-specific wording
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.schemas.system_context import SystemContextBundle

# ── Limits ────────────────────────────────────────────────────────────

MAX_ANSWER_CHARS = 1500
MAX_PERMISSIONS_SHOWN = 6
MAX_METRICS_SHOWN = 6
MAX_RECORDS_SHOWN = 4
MAX_WORKFLOWS_SHOWN = 4

# ── Text helpers ──────────────────────────────────────────────────────

_RE_HTML = re.compile(r"<[^>]+>")
_RE_MULTI_WS = re.compile(r"\s{2,}")


def _clean(text: str | None) -> str:
    if not text:
        return ""
    text = _RE_HTML.sub("", text)
    text = _RE_MULTI_WS.sub(" ", text)
    return text.strip()


def _trunc(text: str, limit: int = 150) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


# ── Public API ────────────────────────────────────────────────────────


def build_system_only_answer(
    *,
    question: str,
    category: str,
    bundle: SystemContextBundle | None,
) -> str | None:
    """Build a deterministic answer from system context data.

    Args:
        question: Original user question (for context, not parsed deeply).
        category: Question category from orchestrator (system/access/mixed/...).
        bundle: The SystemContextBundle with available data.

    Returns:
        A human-readable answer string, or None if no useful data
        is available to construct an answer.
    """
    if bundle is None:
        return None

    if category == "access":
        return _build_access_answer(bundle)
    elif category == "system":
        return _build_system_answer(bundle)
    elif category == "mixed":
        # For mixed fallback, try system first, then access
        answer = _build_system_answer(bundle)
        if answer is None:
            answer = _build_access_answer(bundle)
        return answer
    else:
        # knowledge / unknown — system builder shouldn't be called,
        # but if it is, try to provide whatever we have
        return _build_system_answer(bundle) or _build_access_answer(bundle)


# ── Category-specific builders ────────────────────────────────────────


def _build_access_answer(bundle: SystemContextBundle) -> str | None:
    """Build an answer for ACCESS-category questions.

    Focuses on: user identity, role, tenant, permission decisions.
    """
    parts: list[str] = []

    # User identity
    if bundle.user is not None:
        u = bundle.user
        identity_parts: list[str] = []
        if u.display_name:
            identity_parts.append(f"**{_clean(u.display_name)}**")
        if u.role:
            identity_parts.append(f"role: {_clean(u.role)}")
        if u.roles:
            identity_parts.append(f"roles: {', '.join(u.roles[:5])}")

        if identity_parts:
            parts.append("Current user: " + ", ".join(identity_parts) + ".")

    # Tenant
    if bundle.tenant is not None and bundle.tenant.tenant_name:
        parts.append(f"Organization: {_clean(bundle.tenant.tenant_name)}.")

    # Permissions
    if bundle.has_permissions and bundle.permissions:
        decisions = bundle.permissions.decisions[:MAX_PERMISSIONS_SHOWN]
        if decisions:
            allowed = [d for d in decisions if d.allowed]
            denied = [d for d in decisions if not d.allowed]

            perm_lines: list[str] = []
            if allowed:
                items = [f"{d.resource_type}/{d.action}" for d in allowed]
                perm_lines.append(f"Allowed: {', '.join(items)}.")
            if denied:
                items = [f"{d.resource_type}/{d.action}" for d in denied]
                perm_lines.append(f"Denied: {', '.join(items)}.")

            if perm_lines:
                parts.append("Permission summary:\n" + "\n".join(perm_lines))
        else:
            parts.append(
                "No specific permission decisions are available in the current context. "
                "Please contact your administrator for detailed access information."
            )
    elif bundle.permissions is not None:
        # Permissions exist but empty (fail-closed scenario)
        parts.append(
            "No specific permission decisions are available in the current context. "
            "Access may be restricted by default — please contact your administrator."
        )
    else:
        parts.append(
            "Permission information is not available at this time. "
            "Please contact your administrator for access details."
        )

    if not parts:
        return None

    answer = "\n\n".join(parts)
    return _trunc(answer, MAX_ANSWER_CHARS)


def _build_system_answer(bundle: SystemContextBundle) -> str | None:
    """Build an answer for SYSTEM-category questions.

    Focuses on: tenant stats, workflows, recent records.
    """
    parts: list[str] = []

    # Tenant stats
    if bundle.has_stats and bundle.tenant_stats:
        metrics = bundle.tenant_stats.metrics[:MAX_METRICS_SHOWN]
        if metrics:
            stat_lines: list[str] = []
            for m in metrics:
                label = m.label or m.key
                unit = f" {m.unit}" if m.unit else ""
                stat_lines.append(f"- {_clean(label)}: {m.value}{unit}")
            period = ""
            if bundle.tenant_stats.period:
                period = f" ({_clean(bundle.tenant_stats.period)})"
            parts.append(f"Current statistics{period}:\n" + "\n".join(stat_lines))

    # Workflows
    if bundle.workflows:
        wf_lines: list[str] = []
        for w in bundle.workflows[:MAX_WORKFLOWS_SHOWN]:
            wf_parts = [f"- {_clean(w.workflow_type)}"]
            if w.total is not None:
                wf_parts.append(f"total: {w.total}")
            if w.pending_count is not None:
                wf_parts.append(f"pending: {w.pending_count}")
            if w.completed_count is not None:
                wf_parts.append(f"completed: {w.completed_count}")
            wf_lines.append(", ".join(wf_parts))
        parts.append("Workflow summary:\n" + "\n".join(wf_lines))

    # Recent records
    if bundle.records:
        rec_lines: list[str] = []
        for r in bundle.records[:MAX_RECORDS_SHOWN]:
            title = _clean(r.title) or r.record_id
            status = f" [{_clean(r.status)}]" if r.status else ""
            summary = f" — {_trunc(_clean(r.summary or ''), 80)}" if r.summary else ""
            rec_lines.append(f"- {_clean(r.record_type)} {title}{status}{summary}")
        parts.append("Recent records:\n" + "\n".join(rec_lines))

    if not parts:
        return None

    answer = "\n\n".join(parts)
    return _trunc(answer, MAX_ANSWER_CHARS)
