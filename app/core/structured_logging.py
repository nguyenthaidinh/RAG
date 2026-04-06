"""
Structured JSON logging helper (Phase 6.0).

Provides ``log_event()`` that emits one-line JSON via stdlib logger.
All log output includes mandatory fields for request correlation.

Privacy: NEVER log raw query text, snippets, embeddings, or prompts.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

_json_logger = logging.getLogger("structured")

if not _json_logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    _json_logger.addHandler(_handler)
    _json_logger.setLevel(logging.INFO)


# Safe keys that may appear in structured logs (mirrors audit allowlist).
_LOG_SAFE_KEYS: frozenset[str] = frozenset({
    "request_id", "idempotency_key",
    "results_count", "latency_ms",
    "tokens_query", "tokens_context", "tokens_total",
    "limit", "remaining", "reset_at",
    "plan_code", "override_changed",
    "tenant_id", "user_id",
    "dedupe_key", "time_bucket",
    "reason_code", "http_status",
    "scope", "mode", "override_keys_changed",
    "target_tenant_id", "redacted",
    "event_type", "severity",
})


def log_event(
    event: str,
    *,
    tenant_id: str | None = None,
    user_id: int | None = None,
    request_id: str | None = None,
    idempotency_key: str | None = None,
    status: str = "ok",
    http_status: int | None = None,
    **extra: Any,
) -> None:
    """
    Emit a single-line JSON log entry.

    Mandatory fields are always present (even if ``None``).
    Extra keyword arguments are added only if they are privacy-safe.
    """
    record: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "tenant_id": tenant_id,
        "user_id": user_id,
        "request_id": request_id,
        "status": status,
    }

    if idempotency_key is not None:
        record["idempotency_key"] = idempotency_key
    if http_status is not None:
        record["http_status"] = http_status

    for k, v in extra.items():
        if k in _LOG_SAFE_KEYS:
            record[k] = v

    try:
        _json_logger.info(json.dumps(record, default=str))
    except Exception:
        _json_logger.error(json.dumps({"event": event, "error": "log_serialization_failed"}))
