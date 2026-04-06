"""
FastAPI dependency for the ``X-Idempotency-Key`` request header.

When a query endpoint is exposed (future phase), use this dependency to
extract the client-provided key or generate a UUID fallback.

Client-provided keys are retry-safe; auto-generated keys are not.
"""
from __future__ import annotations

import uuid

from fastapi import Header


def get_idempotency_key(
    x_idempotency_key: str | None = Header(
        None,
        alias="X-Idempotency-Key",
        description="Client-provided idempotency key for retry-safe billing",
    ),
) -> str:
    """
    Return the client-supplied idempotency key, or a random UUID if the
    header is absent.
    """
    return x_idempotency_key or str(uuid.uuid4())
