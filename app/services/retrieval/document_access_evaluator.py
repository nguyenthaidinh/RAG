"""
Document-level ACL evaluator for retrieval (Phase 6).

Enforces access_scope metadata from source platform documents at the
retrieval layer.  Three components:

  1. **normalize_access_scope** — parse raw access_scope dict into a
     validated NormalizedAccessScope
  2. **evaluate_document_access** — decide allow/deny for a single
     document given user context
  3. **filter_candidates_by_acl** — batch filter for retrieval results

Design rules:
  - Deny-by-default for restricted documents
  - Explicit deny beats explicit allow
  - Documents without access_scope → tenant-wide (unchanged)
  - No DB queries — operates purely on metadata dicts
  - No vendor / business logic — generic multi-source
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ── Normalized access scope ──────────────────────────────────────────


@dataclass(frozen=True)
class NormalizedAccessScope:
    """Parsed, validated access scope from document metadata."""

    visibility: str = "tenant"  # "tenant" | "restricted"
    allow_user_ids: tuple[str, ...] = ()
    deny_user_ids: tuple[str, ...] = ()
    role_codes: tuple[str, ...] = ()
    permission_keys: tuple[str, ...] = ()
    is_malformed: bool = False

    @property
    def is_restricted(self) -> bool:
        return self.visibility == "restricted"

    @property
    def has_any_rules(self) -> bool:
        return bool(
            self.allow_user_ids
            or self.deny_user_ids
            or self.role_codes
            or self.permission_keys
        )


# Singleton for tenant-wide (no restriction)
_TENANT_WIDE = NormalizedAccessScope(visibility="tenant")


def normalize_access_scope(
    raw: Any | None,
    *,
    scope_present: bool = False,
) -> NormalizedAccessScope:
    """Parse raw access_scope into a validated NormalizedAccessScope.

    Alias normalization:
      - ``roles`` → ``role_codes``
      - ``permissions`` → ``permission_keys``

    Safety semantics (deny-safe for malformed):
      - ``scope_present=False`` (key absent / None) → tenant-wide
      - ``scope_present=True`` but raw is non-dict → **malformed → deny**
      - ``scope_present=True`` and empty dict → tenant-wide
      - ``visibility`` has unknown value → **malformed → deny**
      - ``visibility="restricted"`` with no rules → **malformed → deny**
      - Non-string list items are coerced via str()
    """
    # ── No scope present → tenant-wide (backward compatible) ─────
    if not scope_present:
        return _TENANT_WIDE

    # ── Scope key exists but wrong type → malformed → deny ───────
    if raw is not None and not isinstance(raw, dict):
        logger.warning(
            "document_acl.invalid_scope_type type=%s",
            type(raw).__name__,
        )
        return NormalizedAccessScope(
            visibility="restricted", is_malformed=True,
        )

    # ── Scope key exists but empty/None → tenant-wide ────────────
    if not raw:
        return _TENANT_WIDE

    # ── Parse visibility ─────────────────────────────────────────
    raw_visibility = raw.get("visibility")
    if raw_visibility is None:
        # dict exists but no visibility key → tenant-wide
        # (e.g. older metadata with only provenance info)
        return _TENANT_WIDE

    visibility = str(raw_visibility).lower().strip()

    if visibility == "tenant":
        return _TENANT_WIDE

    if visibility != "restricted":
        # Unrecognised visibility value → malformed → deny
        logger.warning(
            "document_acl.invalid_visibility value=%s",
            raw_visibility,
        )
        return NormalizedAccessScope(
            visibility="restricted", is_malformed=True,
        )

    # ── Restricted: parse rules ──────────────────────────────────
    allow_user_ids = _to_str_tuple(raw.get("allow_user_ids"))
    deny_user_ids = _to_str_tuple(raw.get("deny_user_ids"))

    # Alias: roles → role_codes
    role_codes = _to_str_tuple(
        raw.get("role_codes") or raw.get("roles")
    )

    # Alias: permissions → permission_keys
    permission_keys = _to_str_tuple(
        raw.get("permission_keys") or raw.get("permissions")
    )

    has_rules = bool(
        allow_user_ids or deny_user_ids or role_codes or permission_keys
    )

    return NormalizedAccessScope(
        visibility="restricted",
        allow_user_ids=allow_user_ids,
        deny_user_ids=deny_user_ids,
        role_codes=role_codes,
        permission_keys=permission_keys,
        is_malformed=not has_rules,
    )


def _to_str_tuple(val: Any) -> tuple[str, ...]:
    """Coerce a value to a tuple of strings."""
    if val is None:
        return ()
    if isinstance(val, (list, tuple, set, frozenset)):
        return tuple(str(v) for v in val if v is not None)
    return ()


# ── User access context ──────────────────────────────────────────────


@dataclass(frozen=True)
class UserAccessContext:
    """Lightweight user context for ACL evaluation.

    Built from the query caller's identity.  When system context is
    available, ``role_codes`` and ``permissions`` are populated from
    ``UserContext.roles`` and ``UserContext.scopes``.
    """

    user_id: str = ""
    role_codes: tuple[str, ...] = ()
    permissions: tuple[str, ...] = ()

    @staticmethod
    def from_query_caller(
        *,
        user_id: int | str,
        role_codes: Sequence[str] | None = None,
        permissions: Sequence[str] | None = None,
    ) -> UserAccessContext:
        """Build from query caller parameters."""
        return UserAccessContext(
            user_id=str(user_id),
            role_codes=tuple(role_codes or ()),
            permissions=tuple(permissions or ()),
        )


# ── Access evaluation ────────────────────────────────────────────────


@dataclass(frozen=True)
class AccessDecision:
    """Result of evaluating document access for a user."""

    allowed: bool
    reason: str


# Pre-built decisions (avoid allocation in hot path)
_ALLOW_TENANT_WIDE = AccessDecision(allowed=True, reason="tenant_wide")
_DENY_NO_CONTEXT = AccessDecision(allowed=False, reason="restricted_no_context")
_DENY_EXPLICIT = AccessDecision(allowed=False, reason="explicit_deny")
_ALLOW_EXPLICIT = AccessDecision(allowed=True, reason="explicit_allow")
_DENY_MALFORMED = AccessDecision(allowed=False, reason="malformed_scope")
_DENY_NO_RULE = AccessDecision(allowed=False, reason="no_matching_rule")


def evaluate_document_access(
    scope: NormalizedAccessScope,
    user: UserAccessContext | None,
) -> AccessDecision:
    """Evaluate whether a user may access a document.

    Evaluation order (first match wins):
      1. Tenant-wide → allow
      2. Restricted + no user context → deny
      3. Malformed restricted → deny
      4. Explicit deny (user_id in deny_user_ids) → deny
      5. Explicit allow (user_id in allow_user_ids) → allow
      6. Role match (ANY of role_codes) → allow
      7. Permission match (ALL of permission_keys) → allow
      8. Default → deny
    """
    # 1. Tenant-wide: no restriction
    if not scope.is_restricted:
        return _ALLOW_TENANT_WIDE

    # 2. Restricted but no user context: deny-by-default
    if user is None or not user.user_id:
        return _DENY_NO_CONTEXT

    # 3. Malformed restricted scope: deny-safe
    if scope.is_malformed:
        return _DENY_MALFORMED

    # 4. Explicit deny: deny_user_ids check
    if scope.deny_user_ids and user.user_id in scope.deny_user_ids:
        return _DENY_EXPLICIT

    # 5. Explicit allow: allow_user_ids check
    if scope.allow_user_ids and user.user_id in scope.allow_user_ids:
        return _ALLOW_EXPLICIT

    # 6. Role check: ANY-match
    if scope.role_codes:
        if user.role_codes and set(user.role_codes).intersection(scope.role_codes):
            return AccessDecision(allowed=True, reason="role_match")
        # Role codes defined but user doesn't match any
        if not scope.permission_keys:
            # No permission_keys to fall through to → deny
            return AccessDecision(allowed=False, reason="role_mismatch")

    # 7. Permission check: ALL-match
    if scope.permission_keys:
        if user.permissions and set(scope.permission_keys).issubset(user.permissions):
            return AccessDecision(allowed=True, reason="permission_match")
        return AccessDecision(allowed=False, reason="missing_permission")

    # 8. Default deny for restricted
    return _DENY_NO_RULE


# ── Batch filter helper ──────────────────────────────────────────────


@dataclass
class AclTraceInfo:
    """Observability info for ACL filtering."""

    total: int = 0
    allowed_count: int = 0
    denied_count: int = 0
    reasons: dict[str, int] = field(default_factory=dict)

    def record(self, decision: AccessDecision) -> None:
        self.total += 1
        if decision.allowed:
            self.allowed_count += 1
        else:
            self.denied_count += 1
        self.reasons[decision.reason] = self.reasons.get(decision.reason, 0) + 1


def extract_access_scope_from_metadata(
    meta: dict[str, Any] | None,
) -> tuple[Any, bool]:
    """Extract access_scope from document metadata.

    Looks in ``metadata.source_platform.access_scope``.

    Returns:
        ``(raw_value, scope_present)`` where:
          - ``scope_present=False`` means the key is absent → tenant-wide
          - ``scope_present=True`` means the key exists; raw_value may
            be a valid dict, an invalid type, empty, etc.

    This distinction is critical for deny-safe behavior: a document
    that never had an access_scope should remain tenant-wide, but a
    document whose access_scope is malformed must be denied.
    """
    if not meta or not isinstance(meta, dict):
        return None, False
    sp = meta.get("source_platform")
    if not sp or not isinstance(sp, dict):
        return None, False
    if "access_scope" not in sp:
        return None, False
    raw = sp["access_scope"]
    # None value = explicitly absent
    if raw is None:
        return None, False
    return raw, True


def filter_candidates_by_acl(
    candidates: list[T],
    *,
    user_ctx: UserAccessContext | None,
    get_metadata: Any,  # Callable[[T], dict | None]
    label: str = "retrieval",
) -> tuple[list[T], AclTraceInfo]:
    """Filter a list of candidates by document-level ACL.

    Args:
        candidates: Items to filter.
        user_ctx: User access context (None = no context available).
        get_metadata: Callable that extracts the document metadata dict
                      from a candidate item.
        label: Label for log messages.

    Returns:
        (filtered_candidates, trace_info)
    """
    if not candidates:
        return candidates, AclTraceInfo()

    trace = AclTraceInfo()
    allowed: list[T] = []

    for item in candidates:
        meta = get_metadata(item)
        raw_scope, scope_present = extract_access_scope_from_metadata(meta)
        scope = normalize_access_scope(raw_scope, scope_present=scope_present)
        decision = evaluate_document_access(scope, user_ctx)
        trace.record(decision)

        if decision.allowed:
            allowed.append(item)
        elif logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "document_acl.denied label=%s reason=%s",
                label,
                decision.reason,
            )

    if trace.denied_count > 0:
        logger.info(
            "document_acl.filtered label=%s total=%d allowed=%d denied=%d reasons=%s",
            label,
            trace.total,
            trace.allowed_count,
            trace.denied_count,
            trace.reasons,
        )

    return allowed, trace


# ── User access context resolver ─────────────────────────────────────


async def resolve_user_access_context(
    *,
    tenant_id: str,
    user_id: int | str,
) -> UserAccessContext:
    """Resolve UserAccessContext for the current user.

    CTDT fork: system context connector removed.
    Returns context with user_id only — sufficient for single-tenant use.
    Role/permission-based document ACL is not used in this deployment.
    """
    return UserAccessContext.from_query_caller(user_id=user_id)
