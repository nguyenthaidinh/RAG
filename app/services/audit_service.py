"""
Audit service (Phase 6.0).

Central entry point to emit privacy-safe, deduplicated audit events.

Key policies:
  - Metadata allowlist: only known-safe keys are stored.
  - Denylist: recursive traversal drops any privacy-sensitive keys.
  - Dedupe: RATE_LIMIT_HIT and TOKEN_QUOTA_EXCEEDED are deduped per
    (tenant_id, user_id, event_type) in minute-buckets.
  - Append-only: the service never updates or deletes events.
  - Failure isolation: audit failures never break the caller (try/except).
"""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from app.core.config import settings
from app.repos.audit_event_repo import (
    AuditEventCreate,
    AuditEventRepository,
    DedupeArgs,
    InMemoryAuditEventRepository,
)

logger = logging.getLogger(__name__)


# ── Event Taxonomy Enums ──────────────────────────────────────────────


class EventType(str, Enum):
    QUERY_EXECUTED = "QUERY_EXECUTED"
    RATE_LIMIT_HIT = "RATE_LIMIT_HIT"
    TOKEN_QUOTA_EXCEEDED = "TOKEN_QUOTA_EXCEEDED"
    PLAN_CHANGED = "PLAN_CHANGED"
    TENANT_QUOTA_OVERRIDE = "TENANT_QUOTA_OVERRIDE"
    ADMIN_ACTION = "ADMIN_ACTION"
    LOGIN_SUCCESS = "LOGIN_SUCCESS"
    LOGIN_FAILURE = "LOGIN_FAILURE"
    DOCUMENT_UPLOADED = "DOCUMENT_UPLOADED"
    DOCUMENT_DELETED = "DOCUMENT_DELETED"


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class Actor(str, Enum):
    SYSTEM = "system"
    ADMIN = "admin"
    USER = "user"


class RefType(str, Enum):
    QUERY = "query"
    QUOTA = "quota"
    RATE_LIMIT = "rate_limit"
    PLAN = "plan"
    ADMIN = "admin"
    AUTH = "auth"
    DOCUMENT = "document"


# ── Policy Mapping ────────────────────────────────────────────────────

EVENT_POLICY: dict[str, dict[str, str]] = {
    EventType.QUERY_EXECUTED: {
        "severity": Severity.INFO,
        "actor": Actor.USER,
        "ref_type": RefType.QUERY,
    },
    EventType.RATE_LIMIT_HIT: {
        "severity": Severity.WARNING,
        "actor": Actor.SYSTEM,
        "ref_type": RefType.RATE_LIMIT,
    },
    EventType.TOKEN_QUOTA_EXCEEDED: {
        "severity": Severity.WARNING,
        "actor": Actor.SYSTEM,
        "ref_type": RefType.QUOTA,
    },
    EventType.PLAN_CHANGED: {
        "severity": Severity.INFO,
        "actor": Actor.ADMIN,
        "ref_type": RefType.PLAN,
    },
    EventType.TENANT_QUOTA_OVERRIDE: {
        "severity": Severity.INFO,
        "actor": Actor.ADMIN,
        "ref_type": RefType.QUOTA,
    },
    EventType.ADMIN_ACTION: {
        "severity": Severity.INFO,
        "actor": Actor.ADMIN,
        "ref_type": RefType.ADMIN,
    },
    EventType.LOGIN_SUCCESS: {
        "severity": Severity.INFO,
        "actor": Actor.USER,
        "ref_type": RefType.AUTH,
    },
    EventType.LOGIN_FAILURE: {
        "severity": Severity.WARNING,
        "actor": Actor.SYSTEM,
        "ref_type": RefType.AUTH,
    },
    EventType.DOCUMENT_UPLOADED: {
        "severity": Severity.INFO,
        "actor": Actor.USER,
        "ref_type": RefType.DOCUMENT,
    },
    EventType.DOCUMENT_DELETED: {
        "severity": Severity.INFO,
        "actor": Actor.USER,
        "ref_type": RefType.DOCUMENT,
    },
}


# ── Privacy: Metadata Allowlist & Denylist ────────────────────────────

METADATA_ALLOWLIST: frozenset[str] = frozenset({
    "request_id",
    "idempotency_key",
    "results_count",
    "latency_ms",
    "tokens_query",
    "tokens_context",
    "tokens_total",
    "limit",
    "remaining",
    "reset_at",
    "plan_code",
    "override_changed",
    "tenant_id",
    "user_id",
    "dedupe_key",
    "time_bucket",
    "reason_code",
    "http_status",
    "scope",
    "mode",
    "override_keys_changed",
    "target_tenant_id",
    "redacted",
    # Auth events (Phase 6 — login)
    "email_hash",       # SHA-256 truncated — NEVER store raw email
    "ip_hash",          # SHA-256 truncated — privacy-safe
    "login_method",     # "api" | "web"
    # Document lifecycle (Phase 6)
    "action",           # "created" | "updated" | "deleted"
    "document_id",
    "source",
    "external_id",
})

METADATA_DENYLIST: frozenset[str] = frozenset({
    "query",
    "query_text",
    "prompt",
    "text",
    "content",
    "snippet",
    "chunks",
    "documents",
    "embedding",
    "vector",
    "context",
    "passage",
    "raw",
    "message",
})


def sanitize_metadata(data: dict[str, Any]) -> dict[str, Any]:
    """
    Return a copy of *data* containing only allowlisted keys.

    Keys in the denylist are dropped.  Keys outside the allowlist are
    also dropped, and ``redacted: True`` is set so callers know data
    was stripped.
    """
    safe: dict[str, Any] = {}
    had_redaction = False

    for key, value in data.items():
        if key in METADATA_DENYLIST:
            had_redaction = True
            continue
        if key in METADATA_ALLOWLIST:
            # Recursively sanitize nested dicts
            if isinstance(value, dict):
                safe[key] = _sanitize_dict_recursive(value)
            else:
                safe[key] = value
        else:
            had_redaction = True

    if had_redaction:
        safe["redacted"] = True

    return safe


def _sanitize_dict_recursive(data: dict[str, Any]) -> dict[str, Any]:
    """Recursively remove denylist keys from nested dicts."""
    result: dict[str, Any] = {}
    for k, v in data.items():
        if k in METADATA_DENYLIST:
            continue
        if isinstance(v, dict):
            result[k] = _sanitize_dict_recursive(v)
        else:
            result[k] = v
    return result


# ── Dedupe Helpers ────────────────────────────────────────────────────

# Event types that require minute-bucket deduplication.
DEDUPED_EVENT_TYPES: frozenset[str] = frozenset({
    EventType.RATE_LIMIT_HIT,
    EventType.TOKEN_QUOTA_EXCEEDED,
})


def build_time_bucket(now: datetime) -> str:
    """Truncate *now* to the minute, format as ``YYYY-MM-DDTHH:MMZ``."""
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%MZ")


def build_dedupe_key(
    tenant_id: str,
    user_id: int | None,
    event_type: str,
    reason_code: str | None = None,
) -> str:
    """SHA-256-based dedupe key (first 16 hex chars)."""
    raw = f"{tenant_id}:{user_id or 'anon'}:{event_type}:{reason_code or ''}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ── Audit Service ─────────────────────────────────────────────────────


class AuditService:
    """
    Central entry point for emitting audit events.

    All helpers validate enums, enforce privacy, and handle dedupe.
    Failures are caught and logged — never propagated to callers.
    """

    __slots__ = ("_repo",)

    def __init__(
        self,
        repo: AuditEventRepository | InMemoryAuditEventRepository | None = None,
    ) -> None:
        self._repo = repo or AuditEventRepository()

    # ── Private core emitter ──────────────────────────────────────

    async def _emit(
        self,
        db: Any,
        *,
        event_type: str,
        tenant_id: str,
        user_id: int | None = None,
        actor: str | None = None,
        severity: str | None = None,
        ref_type: str | None = None,
        ref_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        reason_code: str | None = None,
        now: datetime | None = None,
    ) -> None:
        """
        Core emit logic.  All public helpers delegate here.

        - Resolves defaults from EVENT_POLICY
        - Sanitizes metadata
        - Handles dedupe for RATE_LIMIT_HIT / TOKEN_QUOTA_EXCEEDED
        """
        if not settings.AUDIT_ENABLED:
            return

        if now is None:
            now = datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        policy = EVENT_POLICY.get(event_type, {})
        resolved_actor = actor or policy.get("actor", Actor.SYSTEM)
        resolved_severity = severity or policy.get("severity", Severity.INFO)
        resolved_ref_type = ref_type or policy.get("ref_type")

        safe_meta = sanitize_metadata(metadata or {})

        event = AuditEventCreate(
            event_type=event_type,
            tenant_id=tenant_id,
            user_id=user_id,
            actor=resolved_actor,
            severity=resolved_severity,
            ref_type=resolved_ref_type,
            ref_id=ref_id,
            metadata_json=safe_meta,
        )

        try:
            if event_type in DEDUPED_EVENT_TYPES:
                dedupe = DedupeArgs(
                    dedupe_key=build_dedupe_key(
                        tenant_id, user_id, event_type, reason_code
                    ),
                    time_bucket=build_time_bucket(now),
                )
                await self._repo.create_event_if_absent(db, dedupe, event)
            else:
                await self._repo.create_event(db, event)

            if db is not None and hasattr(db, "commit"):
                await db.commit()

        except Exception:
            logger.warning(
                "audit.emit_failed event_type=%s tenant_id=%s",
                event_type,
                tenant_id,
                exc_info=True,
            )

    # ── Public Helpers ────────────────────────────────────────────

    async def log_query_executed(
        self,
        db: Any,
        *,
        tenant_id: str,
        user_id: int | None = None,
        request_id: str | None = None,
        idempotency_key: str | None = None,
        results_count: int = 0,
        latency_ms: int = 0,
        tokens_query: int = 0,
        tokens_context: int = 0,
        tokens_total: int = 0,
        mode: str | None = None,
    ) -> None:
        """Emit QUERY_EXECUTED event after a successful retrieval."""
        await self._emit(
            db,
            event_type=EventType.QUERY_EXECUTED,
            tenant_id=tenant_id,
            user_id=user_id,
            metadata={
                "request_id": request_id,
                "idempotency_key": idempotency_key,
                "results_count": results_count,
                "latency_ms": latency_ms,
                "tokens_query": tokens_query,
                "tokens_context": tokens_context,
                "tokens_total": tokens_total,
                "mode": mode,
            },
        )

    async def log_rate_limit_hit(
        self,
        db: Any,
        *,
        tenant_id: str,
        user_id: int | None = None,
        request_id: str | None = None,
        limit: int = 0,
        remaining: int = 0,
        reset_at: str | None = None,
        scope: str | None = None,
        reason_code: str = "rate_limit_exceeded",
        now: datetime | None = None,
    ) -> None:
        """Emit RATE_LIMIT_HIT event (deduped per minute-bucket)."""
        await self._emit(
            db,
            event_type=EventType.RATE_LIMIT_HIT,
            tenant_id=tenant_id,
            user_id=user_id,
            reason_code=reason_code,
            now=now,
            metadata={
                "request_id": request_id,
                "limit": limit,
                "remaining": remaining,
                "reset_at": reset_at,
                "scope": scope,
                "reason_code": reason_code,
            },
        )

    async def log_quota_exceeded(
        self,
        db: Any,
        *,
        tenant_id: str,
        user_id: int | None = None,
        request_id: str | None = None,
        reason_code: str = "quota_exceeded",
        tokens_total: int = 0,
        limit: int | None = None,
        http_status: int = 429,
        now: datetime | None = None,
    ) -> None:
        """Emit TOKEN_QUOTA_EXCEEDED event (deduped per minute-bucket)."""
        await self._emit(
            db,
            event_type=EventType.TOKEN_QUOTA_EXCEEDED,
            tenant_id=tenant_id,
            user_id=user_id,
            reason_code=reason_code,
            now=now,
            metadata={
                "request_id": request_id,
                "reason_code": reason_code,
                "tokens_total": tokens_total,
                "limit": limit,
                "http_status": http_status,
            },
        )

    async def log_plan_changed(
        self,
        db: Any,
        *,
        tenant_id: str,
        user_id: int | None = None,
        request_id: str | None = None,
        plan_code: str = "",
        target_tenant_id: str | None = None,
    ) -> None:
        """Emit PLAN_CHANGED event when a plan is created or updated."""
        await self._emit(
            db,
            event_type=EventType.PLAN_CHANGED,
            tenant_id=tenant_id,
            user_id=user_id,
            metadata={
                "request_id": request_id,
                "plan_code": plan_code,
                "target_tenant_id": target_tenant_id,
            },
        )

    async def log_tenant_quota_override(
        self,
        db: Any,
        *,
        tenant_id: str,
        user_id: int | None = None,
        request_id: str | None = None,
        plan_code: str | None = None,
        override_changed: bool = False,
        override_keys_changed: list[str] | None = None,
        target_tenant_id: str | None = None,
    ) -> None:
        """Emit TENANT_QUOTA_OVERRIDE when tenant settings are updated."""
        await self._emit(
            db,
            event_type=EventType.TENANT_QUOTA_OVERRIDE,
            tenant_id=tenant_id,
            user_id=user_id,
            metadata={
                "request_id": request_id,
                "plan_code": plan_code,
                "override_changed": override_changed,
                "override_keys_changed": override_keys_changed,
                "target_tenant_id": target_tenant_id,
            },
        )

    async def log_admin_action(
        self,
        db: Any,
        *,
        tenant_id: str,
        user_id: int | None = None,
        request_id: str | None = None,
        reason_code: str | None = None,
        **extra: Any,
    ) -> None:
        """Emit a generic ADMIN_ACTION event."""
        meta = {"request_id": request_id, "reason_code": reason_code}
        meta.update(extra)
        await self._emit(
            db,
            event_type=EventType.ADMIN_ACTION,
            tenant_id=tenant_id,
            user_id=user_id,
            metadata=meta,
        )

    async def log_login_success(
        self,
        db: Any,
        *,
        tenant_id: str,
        user_id: int | None = None,
        request_id: str | None = None,
        email_hash: str | None = None,
        ip_hash: str | None = None,
        login_method: str = "api",
    ) -> None:
        """Emit LOGIN_SUCCESS event. Privacy: only hashed identifiers."""
        await self._emit(
            db,
            event_type=EventType.LOGIN_SUCCESS,
            tenant_id=tenant_id,
            user_id=user_id,
            metadata={
                "request_id": request_id,
                "email_hash": email_hash,
                "ip_hash": ip_hash,
                "login_method": login_method,
            },
        )

    async def log_login_failure(
        self,
        db: Any,
        *,
        tenant_id: str = "_unknown",
        request_id: str | None = None,
        email_hash: str | None = None,
        ip_hash: str | None = None,
        login_method: str = "api",
        reason_code: str = "invalid_credentials",
    ) -> None:
        """Emit LOGIN_FAILURE event. Privacy: only hashed identifiers, no raw email."""
        await self._emit(
            db,
            event_type=EventType.LOGIN_FAILURE,
            tenant_id=tenant_id,
            severity=Severity.WARNING,
            metadata={
                "request_id": request_id,
                "email_hash": email_hash,
                "ip_hash": ip_hash,
                "login_method": login_method,
                "reason_code": reason_code,
            },
        )

    async def log_document_uploaded(
        self,
        db: Any,
        *,
        tenant_id: str,
        user_id: int | None = None,
        request_id: str | None = None,
        document_id: int | None = None,
        action: str = "created",
        source: str | None = None,
        external_id: str | None = None,
    ) -> None:
        """Emit DOCUMENT_UPLOADED event. Privacy: no raw text/content."""
        await self._emit(
            db,
            event_type=EventType.DOCUMENT_UPLOADED,
            tenant_id=tenant_id,
            user_id=user_id,
            ref_type=RefType.DOCUMENT,
            ref_id=str(document_id) if document_id else None,
            metadata={
                "request_id": request_id,
                "document_id": document_id,
                "action": action,
                "source": source,
                "external_id": external_id,
            },
        )

    async def log_document_deleted(
        self,
        db: Any,
        *,
        tenant_id: str,
        user_id: int | None = None,
        request_id: str | None = None,
        document_id: int | None = None,
    ) -> None:
        """Emit DOCUMENT_DELETED event."""
        await self._emit(
            db,
            event_type=EventType.DOCUMENT_DELETED,
            tenant_id=tenant_id,
            user_id=user_id,
            ref_type=RefType.DOCUMENT,
            ref_id=str(document_id) if document_id else None,
            metadata={
                "request_id": request_id,
                "document_id": document_id,
                "action": "deleted",
            },
        )


# ── Module-level default (singleton) ──────────────────────────────────

_default_instance: AuditService | None = None


def get_audit_service() -> AuditService:
    """Return the process-wide AuditService singleton."""
    global _default_instance
    if _default_instance is None:
        _default_instance = AuditService()
    return _default_instance
