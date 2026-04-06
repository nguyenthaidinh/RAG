"""
Document event emitter (Phase 2A).

Thin helper to emit document lifecycle events to the document_events table.
Designed to be called from ingest services — always fail-open so document
processing is never blocked by event emission failures.

Design rules:
  - Sanitize metadata_json BEFORE writing — never trust caller
  - Always fail-open (catch + warn)
  - Tenant-scoped
  - No direct DB session management — uses caller's session or creates own
"""
from __future__ import annotations

import logging
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# ── Event type constants ──────────────────────────────────────────────

DOCUMENT_CREATED = "document.created"
DOCUMENT_UPDATED = "document.updated"
DOCUMENT_STATUS_CHANGED = "document.status_changed"
DOCUMENT_SYNTHESIZED_CHILD_CREATED = "document.synthesized_child_created"
DOCUMENT_OPENED = "document.opened"
DOCUMENT_DOWNLOADED = "document.downloaded"

# ── Metadata sanitization ────────────────────────────────────────────

# Keys that might contain raw text / content and MUST be stripped.
# Applied recursively to nested dicts.
_UNSAFE_KEYS = frozenset({
    "content",
    "text",
    "raw_content",
    "content_raw",
    "content_text",
    "chunk_text",
    "prompt",
    "context",
    "body",
    "response_text",
    "answer",
})


def _sanitize_metadata(meta: dict | None) -> dict | None:
    """
    Recursively strip unsafe keys from event metadata.

    Returns a new dict (never mutates the original).
    Returns None if input is None.
    """
    if meta is None:
        return None
    if not isinstance(meta, dict):
        return None

    clean: dict = {}
    for key, value in meta.items():
        if key in _UNSAFE_KEYS:
            continue
        if isinstance(value, dict):
            clean[key] = _sanitize_metadata(value)
        elif isinstance(value, list):
            clean[key] = [
                _sanitize_metadata(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            clean[key] = value
    return clean


# ── Emit functions ────────────────────────────────────────────────────


async def emit_document_event(
    db: AsyncSession,
    *,
    tenant_id: str,
    document_id: int,
    event_type: str,
    from_status: str | None = None,
    to_status: str | None = None,
    actor_user_id: int | None = None,
    request_id: str | None = None,
    message: str | None = None,
    metadata_json: dict | None = None,
) -> None:
    """
    Emit a document lifecycle event.

    metadata_json is sanitized before persisting — unsafe keys are stripped.
    Always fail-open — errors are caught and logged as warnings.
    Never raises.
    """
    # DocumentEvent model removed in CTDT fork — events are logged only
    logger.debug(
        "document_event tenant_id=%s doc_id=%s event_type=%s from=%s to=%s",
        tenant_id, document_id, event_type, from_status, to_status,
    )


async def emit_document_event_standalone(
    *,
    tenant_id: str,
    document_id: int,
    event_type: str,
    from_status: str | None = None,
    to_status: str | None = None,
    actor_user_id: int | None = None,
    request_id: str | None = None,
    message: str | None = None,
    metadata_json: dict | None = None,
) -> None:
    """
    Emit a document event using a standalone session.

    Used for cases where the caller doesn't have an active session
    (e.g., from web routes for open/download logging).
    Always fail-open.
    """
    try:
        from app.db.session import AsyncSessionLocal

        async with AsyncSessionLocal() as db:
            await emit_document_event(
                db,
                tenant_id=tenant_id,
                document_id=document_id,
                event_type=event_type,
                from_status=from_status,
                to_status=to_status,
                actor_user_id=actor_user_id,
                request_id=request_id,
                message=message,
                metadata_json=metadata_json,
            )
            await db.commit()
    except Exception:
        logger.warning(
            "document_event.standalone_emit_failed tenant_id=%s doc_id=%s event_type=%s",
            tenant_id, document_id, event_type,
            exc_info=True,
        )
