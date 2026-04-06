"""
Structured audit logger for security-critical actions.

Every audit entry includes:
  actor_user_id, tenant_id, action, target_id, timestamp, detail
"""

import logging
from datetime import datetime, timezone

audit_logger = logging.getLogger("audit")

# Ensure the audit logger has at least a StreamHandler so messages aren't lost
if not audit_logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter(
            "%(asctime)s AUDIT %(levelname)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
    )
    audit_logger.addHandler(_handler)
    audit_logger.setLevel(logging.INFO)


def audit_log(
    *,
    action: str,
    actor_user_id: int | None = None,
    tenant_id: str | None = None,
    target_id: str | int | None = None,
    detail: str = "",
) -> None:
    """
    Emit a structured audit log entry.

    Parameters
    ----------
    action : str
        e.g. "api_key.create", "api_key.rotate", "api_key.revoke",
             "quota.exceeded", "user.create", "user.role_change"
    actor_user_id : int | None
        The user performing the action.
    tenant_id : str | None
        The tenant context.
    target_id : str | int | None
        The resource being acted upon (e.g. api_key id, user id).
    detail : str
        Human-readable detail about the action.
    """
    ts = datetime.now(timezone.utc).isoformat()
    audit_logger.info(
        "action=%s actor_user_id=%s tenant_id=%s target_id=%s ts=%s detail=%s",
        action,
        actor_user_id,
        tenant_id,
        target_id,
        ts,
        detail,
    )
