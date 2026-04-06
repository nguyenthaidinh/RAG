"""
CSRF protection — double-submit cookie pattern (Phase 8 security hardening).

How it works:
  1. On login, a random ``csrf_token`` cookie is set (JS-readable, NOT HttpOnly).
  2. Every admin POST form includes a hidden field ``csrf_token`` whose value
     is read from the cookie by a small JS snippet in the layout template.
  3. On POST, the server compares the cookie value to the form field value.
     If they don't match (or either is missing), the request is rejected with 403.

Why double-submit cookie:
  - No server-side session storage required (stateless).
  - SameSite=Lax prevents the cookie from being sent on cross-site POST.
  - An attacker on another origin cannot read the cookie (same-origin policy),
    so they cannot forge the hidden field value.

Rules:
  - Login POST is exempt (no cookie exists yet).
  - Only admin POST routes are protected (GET routes are read-only / safe).
  - Validation is server-side only.
"""
from __future__ import annotations

import hmac
import logging
import secrets

from fastapi import HTTPException, Request

logger = logging.getLogger(__name__)


def validate_csrf(request: Request, form_token: str | None) -> None:
    """
    Compare the ``csrf_token`` cookie against the form-submitted value.

    Raises HTTPException(403) on mismatch or missing values.
    """
    cookie_token = request.cookies.get("csrf_token")

    if not cookie_token or not form_token:
        logger.warning("csrf.missing cookie=%s form=%s", bool(cookie_token), bool(form_token))
        raise HTTPException(status_code=403, detail="CSRF validation failed")

    # Constant-time comparison to prevent timing attacks
    if not hmac.compare_digest(cookie_token, form_token):
        logger.warning("csrf.mismatch")
        raise HTTPException(status_code=403, detail="CSRF validation failed")


def generate_csrf_token() -> str:
    """Generate a new CSRF token (URL-safe, 32 bytes entropy)."""
    return secrets.token_urlsafe(32)
