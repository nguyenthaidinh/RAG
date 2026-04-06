"""
Phase 6 — Document access evaluator unit tests.

Covers:
  1. Access scope normalization (empty, tenant-wide, restricted, aliases)
  2. Malformed scope deny-safety (non-dict, unknown visibility, etc.)
  3. Access evaluation (deny-by-default, precedence rules, role/perm matching)
  4. Extract access scope from metadata (tuple returns)
  5. Batch filter helper
  6. User access context construction
  7. resolve_user_access_context (with mocked connector)
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.retrieval.document_access_evaluator import (
    AccessDecision,
    AclTraceInfo,
    NormalizedAccessScope,
    UserAccessContext,
    evaluate_document_access,
    extract_access_scope_from_metadata,
    filter_candidates_by_acl,
    normalize_access_scope,
    resolve_user_access_context,
)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# =====================================================================
# 1. NORMALIZATION — basic behavior
# =====================================================================


class TestNormalizeAccessScope:
    """Test normalize_access_scope parsing and alias handling."""

    # ── No scope present → tenant-wide (backward compatible) ─────

    def test_scope_not_present_returns_tenant_wide(self):
        scope = normalize_access_scope(None, scope_present=False)
        assert scope.visibility == "tenant"
        assert not scope.is_restricted

    def test_scope_not_present_even_with_value(self):
        """When scope_present=False, raw value is ignored."""
        scope = normalize_access_scope({"visibility": "restricted"}, scope_present=False)
        assert scope.visibility == "tenant"

    # ── Scope present, empty dict → tenant-wide ──────────────────

    def test_present_empty_dict_returns_tenant_wide(self):
        scope = normalize_access_scope({}, scope_present=True)
        assert scope.visibility == "tenant"

    def test_present_none_returns_tenant_wide(self):
        scope = normalize_access_scope(None, scope_present=True)
        assert scope.visibility == "tenant"

    # ── Scope present, valid dict ────────────────────────────────

    def test_explicit_tenant_visibility(self):
        scope = normalize_access_scope(
            {"visibility": "tenant"}, scope_present=True,
        )
        assert scope.visibility == "tenant"
        assert not scope.is_restricted

    def test_restricted_with_allow_user_ids(self):
        scope = normalize_access_scope({
            "visibility": "restricted",
            "allow_user_ids": ["10", "20"],
        }, scope_present=True)
        assert scope.is_restricted
        assert scope.allow_user_ids == ("10", "20")
        assert not scope.is_malformed

    def test_restricted_with_deny_user_ids(self):
        scope = normalize_access_scope({
            "visibility": "restricted",
            "deny_user_ids": ["99"],
        }, scope_present=True)
        assert scope.is_restricted
        assert scope.deny_user_ids == ("99",)

    def test_alias_roles_to_role_codes(self):
        scope = normalize_access_scope({
            "visibility": "restricted",
            "roles": ["admin", "editor"],
        }, scope_present=True)
        assert scope.role_codes == ("admin", "editor")
        assert not scope.is_malformed

    def test_alias_permissions_to_permission_keys(self):
        scope = normalize_access_scope({
            "visibility": "restricted",
            "permissions": ["doc.read", "doc.write"],
        }, scope_present=True)
        assert scope.permission_keys == ("doc.read", "doc.write")

    def test_role_codes_takes_precedence_over_roles_alias(self):
        scope = normalize_access_scope({
            "visibility": "restricted",
            "role_codes": ["admin"],
            "roles": ["editor"],  # should be ignored
        }, scope_present=True)
        assert scope.role_codes == ("admin",)

    def test_malformed_restricted_no_rules(self):
        """Restricted with no rules → malformed."""
        scope = normalize_access_scope(
            {"visibility": "restricted"}, scope_present=True,
        )
        assert scope.is_restricted
        assert scope.is_malformed
        assert not scope.has_any_rules

    def test_integer_user_ids_coerced_to_string(self):
        scope = normalize_access_scope({
            "visibility": "restricted",
            "allow_user_ids": [10, 20],
        }, scope_present=True)
        assert scope.allow_user_ids == ("10", "20")

    def test_none_items_in_lists_filtered(self):
        scope = normalize_access_scope({
            "visibility": "restricted",
            "allow_user_ids": [None, "10", None],
        }, scope_present=True)
        assert scope.allow_user_ids == ("10",)

    def test_dict_with_no_visibility_key_returns_tenant_wide(self):
        """Dict exists but has no visibility key → tenant-wide."""
        scope = normalize_access_scope(
            {"some_other_key": "value"}, scope_present=True,
        )
        assert scope.visibility == "tenant"


# =====================================================================
# 2. MALFORMED SCOPE DENY-SAFETY (Hotfix)
# =====================================================================


class TestMalformedScopeDenySafe:
    """Test that malformed/invalid access_scope → DENY, not tenant-wide."""

    def test_non_dict_scope_is_malformed(self):
        """access_scope is a string instead of dict → malformed → deny."""
        scope = normalize_access_scope("restricted", scope_present=True)
        assert scope.is_restricted
        assert scope.is_malformed

    def test_list_scope_is_malformed(self):
        """access_scope is a list instead of dict → malformed → deny."""
        scope = normalize_access_scope([1, 2, 3], scope_present=True)
        assert scope.is_restricted
        assert scope.is_malformed

    def test_integer_scope_is_malformed(self):
        """access_scope is an integer instead of dict → malformed → deny."""
        scope = normalize_access_scope(42, scope_present=True)
        assert scope.is_restricted
        assert scope.is_malformed

    def test_boolean_scope_is_malformed(self):
        """access_scope is a boolean → malformed → deny."""
        scope = normalize_access_scope(True, scope_present=True)
        assert scope.is_restricted
        assert scope.is_malformed

    def test_unknown_visibility_is_malformed(self):
        """Unknown visibility value → malformed → deny."""
        scope = normalize_access_scope(
            {"visibility": "unknown_value"}, scope_present=True,
        )
        assert scope.is_restricted
        assert scope.is_malformed

    def test_visibility_private_is_malformed(self):
        """visibility='private' → not recognized → malformed."""
        scope = normalize_access_scope(
            {"visibility": "private"}, scope_present=True,
        )
        assert scope.is_restricted
        assert scope.is_malformed

    def test_malformed_scope_evaluates_to_deny(self):
        """End-to-end: malformed scope + user → deny."""
        scope = normalize_access_scope("bad", scope_present=True)
        user = UserAccessContext.from_query_caller(user_id=1)
        decision = evaluate_document_access(scope, user)
        assert decision.allowed is False
        assert decision.reason == "malformed_scope"

    def test_unknown_visibility_evaluates_to_deny(self):
        """End-to-end: unknown visibility + user → deny."""
        scope = normalize_access_scope(
            {"visibility": "public"}, scope_present=True,
        )
        user = UserAccessContext.from_query_caller(user_id=1)
        decision = evaluate_document_access(scope, user)
        assert decision.allowed is False
        assert decision.reason == "malformed_scope"


# =====================================================================
# 3. EVALUATION
# =====================================================================


class TestEvaluateDocumentAccess:
    """Test evaluate_document_access decision logic."""

    def test_tenant_wide_allows_anyone(self):
        scope = NormalizedAccessScope(visibility="tenant")
        decision = evaluate_document_access(scope, None)
        assert decision.allowed is True
        assert decision.reason == "tenant_wide"

    def test_tenant_wide_allows_with_user(self):
        scope = NormalizedAccessScope(visibility="tenant")
        user = UserAccessContext.from_query_caller(user_id=1)
        decision = evaluate_document_access(scope, user)
        assert decision.allowed is True

    def test_restricted_no_user_context_denied(self):
        scope = normalize_access_scope({
            "visibility": "restricted",
            "allow_user_ids": ["10"],
        }, scope_present=True)
        decision = evaluate_document_access(scope, None)
        assert decision.allowed is False
        assert decision.reason == "restricted_no_context"

    def test_restricted_empty_user_id_denied(self):
        scope = normalize_access_scope({
            "visibility": "restricted",
            "allow_user_ids": ["10"],
        }, scope_present=True)
        user = UserAccessContext(user_id="")
        decision = evaluate_document_access(scope, user)
        assert decision.allowed is False
        assert decision.reason == "restricted_no_context"

    def test_malformed_restricted_denied(self):
        scope = normalize_access_scope(
            {"visibility": "restricted"}, scope_present=True,
        )
        user = UserAccessContext.from_query_caller(user_id=1)
        decision = evaluate_document_access(scope, user)
        assert decision.allowed is False
        assert decision.reason == "malformed_scope"

    def test_explicit_deny_beats_explicit_allow(self):
        """Deny_user_ids takes precedence over allow_user_ids."""
        scope = normalize_access_scope({
            "visibility": "restricted",
            "allow_user_ids": ["10"],
            "deny_user_ids": ["10"],
        }, scope_present=True)
        user = UserAccessContext.from_query_caller(user_id=10)
        decision = evaluate_document_access(scope, user)
        assert decision.allowed is False
        assert decision.reason == "explicit_deny"

    def test_explicit_allow(self):
        scope = normalize_access_scope({
            "visibility": "restricted",
            "allow_user_ids": ["10", "20"],
        }, scope_present=True)
        user = UserAccessContext.from_query_caller(user_id=10)
        decision = evaluate_document_access(scope, user)
        assert decision.allowed is True
        assert decision.reason == "explicit_allow"

    def test_explicit_allow_not_matched(self):
        scope = normalize_access_scope({
            "visibility": "restricted",
            "allow_user_ids": ["10"],
        }, scope_present=True)
        user = UserAccessContext.from_query_caller(user_id=99)
        decision = evaluate_document_access(scope, user)
        assert decision.allowed is False

    def test_role_codes_any_match(self):
        """User needs ANY of the role_codes to match."""
        scope = normalize_access_scope({
            "visibility": "restricted",
            "role_codes": ["admin", "editor"],
        }, scope_present=True)
        user = UserAccessContext.from_query_caller(
            user_id=1, role_codes=["editor"],
        )
        decision = evaluate_document_access(scope, user)
        assert decision.allowed is True
        assert decision.reason == "role_match"

    def test_role_codes_no_match(self):
        scope = normalize_access_scope({
            "visibility": "restricted",
            "role_codes": ["admin"],
        }, scope_present=True)
        user = UserAccessContext.from_query_caller(
            user_id=1, role_codes=["viewer"],
        )
        decision = evaluate_document_access(scope, user)
        assert decision.allowed is False
        assert decision.reason == "role_mismatch"

    def test_role_codes_no_user_roles(self):
        scope = normalize_access_scope({
            "visibility": "restricted",
            "role_codes": ["admin"],
        }, scope_present=True)
        user = UserAccessContext.from_query_caller(user_id=1)
        decision = evaluate_document_access(scope, user)
        assert decision.allowed is False

    def test_permission_keys_all_match(self):
        """User must have ALL permission_keys."""
        scope = normalize_access_scope({
            "visibility": "restricted",
            "permission_keys": ["doc.read", "doc.export"],
        }, scope_present=True)
        user = UserAccessContext.from_query_caller(
            user_id=1,
            permissions=["doc.read", "doc.export", "doc.write"],
        )
        decision = evaluate_document_access(scope, user)
        assert decision.allowed is True
        assert decision.reason == "permission_match"

    def test_permission_keys_partial_match_denied(self):
        """User has only some of the required permissions → deny."""
        scope = normalize_access_scope({
            "visibility": "restricted",
            "permission_keys": ["doc.read", "doc.export"],
        }, scope_present=True)
        user = UserAccessContext.from_query_caller(
            user_id=1,
            permissions=["doc.read"],
        )
        decision = evaluate_document_access(scope, user)
        assert decision.allowed is False
        assert decision.reason == "missing_permission"

    def test_role_fallthrough_to_permission(self):
        """If role doesn't match but permission_keys are also defined, check those."""
        scope = normalize_access_scope({
            "visibility": "restricted",
            "role_codes": ["admin"],
            "permission_keys": ["doc.read"],
        }, scope_present=True)
        user = UserAccessContext.from_query_caller(
            user_id=1,
            role_codes=["viewer"],
            permissions=["doc.read"],
        )
        decision = evaluate_document_access(scope, user)
        assert decision.allowed is True
        assert decision.reason == "permission_match"


# =====================================================================
# 4. EXTRACT ACCESS SCOPE FROM METADATA
# =====================================================================


class TestExtractAccessScope:
    """Test extract_access_scope_from_metadata helper — returns (raw, present)."""

    def test_none_metadata(self):
        raw, present = extract_access_scope_from_metadata(None)
        assert raw is None
        assert present is False

    def test_empty_metadata(self):
        raw, present = extract_access_scope_from_metadata({})
        assert raw is None
        assert present is False

    def test_no_source_platform(self):
        raw, present = extract_access_scope_from_metadata({"other": True})
        assert present is False

    def test_source_platform_no_access_scope_key(self):
        raw, present = extract_access_scope_from_metadata({
            "source_platform": {"source_key": "test"},
        })
        assert present is False

    def test_access_scope_is_none(self):
        raw, present = extract_access_scope_from_metadata({
            "source_platform": {"access_scope": None},
        })
        assert present is False

    def test_valid_access_scope(self):
        meta = {
            "source_platform": {
                "access_scope": {
                    "visibility": "restricted",
                    "allow_user_ids": ["10"],
                },
            },
        }
        raw, present = extract_access_scope_from_metadata(meta)
        assert present is True
        assert raw["visibility"] == "restricted"
        assert raw["allow_user_ids"] == ["10"]

    def test_access_scope_wrong_type_still_present(self):
        """If access_scope is a string, it's present but invalid."""
        raw, present = extract_access_scope_from_metadata({
            "source_platform": {"access_scope": "bad_value"},
        })
        assert present is True
        assert raw == "bad_value"

    def test_access_scope_list_still_present(self):
        """If access_scope is a list, it's present but invalid."""
        raw, present = extract_access_scope_from_metadata({
            "source_platform": {"access_scope": [1, 2]},
        })
        assert present is True
        assert raw == [1, 2]

    def test_access_scope_empty_dict_is_present(self):
        """Empty dict is still present (→ tenant-wide by normalize)."""
        raw, present = extract_access_scope_from_metadata({
            "source_platform": {"access_scope": {}},
        })
        assert present is True
        assert raw == {}


# =====================================================================
# 5. BATCH FILTER HELPER
# =====================================================================


class _FakeCandidate:
    """Minimal candidate for testing filter_candidates_by_acl."""

    def __init__(self, doc_id: int, metadata: dict | None = None):
        self.document_id = doc_id
        self.metadata = metadata or {}


class TestFilterCandidatesByAcl:
    """Test filter_candidates_by_acl batch filtering."""

    def test_empty_candidates(self):
        filtered, trace = filter_candidates_by_acl(
            [],
            user_ctx=None,
            get_metadata=lambda c: c.metadata,
        )
        assert filtered == []
        assert trace.total == 0

    def test_all_tenant_wide_allowed(self):
        cands = [
            _FakeCandidate(1, {}),
            _FakeCandidate(2, {"source_platform": {"source_key": "test"}}),
        ]
        user = UserAccessContext.from_query_caller(user_id=1)
        filtered, trace = filter_candidates_by_acl(
            cands,
            user_ctx=user,
            get_metadata=lambda c: c.metadata,
        )
        assert len(filtered) == 2
        assert trace.allowed_count == 2
        assert trace.denied_count == 0

    def test_restricted_doc_denied_without_context(self):
        cands = [
            _FakeCandidate(1, {}),
            _FakeCandidate(2, {
                "source_platform": {
                    "access_scope": {
                        "visibility": "restricted",
                        "allow_user_ids": ["99"],
                    },
                },
            }),
        ]
        filtered, trace = filter_candidates_by_acl(
            cands,
            user_ctx=None,
            get_metadata=lambda c: c.metadata,
        )
        assert len(filtered) == 1
        assert filtered[0].document_id == 1
        assert trace.denied_count == 1

    def test_restricted_doc_allowed_for_matching_user(self):
        cands = [
            _FakeCandidate(1, {
                "source_platform": {
                    "access_scope": {
                        "visibility": "restricted",
                        "allow_user_ids": ["42"],
                    },
                },
            }),
        ]
        user = UserAccessContext.from_query_caller(user_id=42)
        filtered, trace = filter_candidates_by_acl(
            cands,
            user_ctx=user,
            get_metadata=lambda c: c.metadata,
        )
        assert len(filtered) == 1
        assert trace.allowed_count == 1

    def test_mixed_tenant_and_restricted(self):
        cands = [
            _FakeCandidate(1, {}),  # tenant-wide
            _FakeCandidate(2, {
                "source_platform": {
                    "access_scope": {
                        "visibility": "restricted",
                        "allow_user_ids": ["10"],
                    },
                },
            }),
            _FakeCandidate(3, {
                "source_platform": {
                    "access_scope": {
                        "visibility": "restricted",
                        "allow_user_ids": ["99"],
                    },
                },
            }),
        ]
        user = UserAccessContext.from_query_caller(user_id=10)
        filtered, trace = filter_candidates_by_acl(
            cands,
            user_ctx=user,
            get_metadata=lambda c: c.metadata,
        )
        assert len(filtered) == 2  # doc 1 (tenant) + doc 2 (allowed)
        assert {c.document_id for c in filtered} == {1, 2}
        assert trace.denied_count == 1

    def test_malformed_scope_denied_in_batch(self):
        """Malformed access_scope (wrong type) → denied in batch filter."""
        cands = [
            _FakeCandidate(1, {}),  # tenant-wide
            _FakeCandidate(2, {
                "source_platform": {
                    "access_scope": "not_a_dict",
                },
            }),
        ]
        user = UserAccessContext.from_query_caller(user_id=1)
        filtered, trace = filter_candidates_by_acl(
            cands,
            user_ctx=user,
            get_metadata=lambda c: c.metadata,
        )
        assert len(filtered) == 1
        assert filtered[0].document_id == 1
        assert trace.denied_count == 1

    def test_role_based_access_in_batch(self):
        """Role-restricted doc allowed when user has matching role."""
        cands = [
            _FakeCandidate(1, {
                "source_platform": {
                    "access_scope": {
                        "visibility": "restricted",
                        "role_codes": ["editor"],
                    },
                },
            }),
        ]
        user = UserAccessContext.from_query_caller(
            user_id=1, role_codes=["editor"],
        )
        filtered, trace = filter_candidates_by_acl(
            cands,
            user_ctx=user,
            get_metadata=lambda c: c.metadata,
        )
        assert len(filtered) == 1
        assert trace.allowed_count == 1

    def test_permission_based_access_denied_in_batch(self):
        """Permission-restricted doc denied when user lacks permissions."""
        cands = [
            _FakeCandidate(1, {
                "source_platform": {
                    "access_scope": {
                        "visibility": "restricted",
                        "permission_keys": ["doc.admin"],
                    },
                },
            }),
        ]
        user = UserAccessContext.from_query_caller(
            user_id=1, permissions=["doc.read"],
        )
        filtered, trace = filter_candidates_by_acl(
            cands,
            user_ctx=user,
            get_metadata=lambda c: c.metadata,
        )
        assert len(filtered) == 0
        assert trace.denied_count == 1


# =====================================================================
# 6. USER ACCESS CONTEXT
# =====================================================================


class TestUserAccessContext:
    """Test UserAccessContext construction."""

    def test_from_query_caller_minimal(self):
        ctx = UserAccessContext.from_query_caller(user_id=42)
        assert ctx.user_id == "42"
        assert ctx.role_codes == ()
        assert ctx.permissions == ()

    def test_from_query_caller_full(self):
        ctx = UserAccessContext.from_query_caller(
            user_id="u-100",
            role_codes=["admin", "editor"],
            permissions=["doc.read"],
        )
        assert ctx.user_id == "u-100"
        assert ctx.role_codes == ("admin", "editor")
        assert ctx.permissions == ("doc.read",)

    def test_from_query_caller_none_lists(self):
        ctx = UserAccessContext.from_query_caller(
            user_id=1,
            role_codes=None,
            permissions=None,
        )
        assert ctx.role_codes == ()
        assert ctx.permissions == ()


# =====================================================================
# 7. resolve_user_access_context (WITH MOCKED CONNECTOR)
# =====================================================================


class TestResolveUserAccessContext:
    """Test resolve_user_access_context loads roles/permissions from system context."""

    def test_returns_full_context_when_connector_available(self):
        """Should extract roles and scopes from UserContext."""
        from app.schemas.system_context import UserContext

        mock_user = UserContext(
            user_id=10,
            tenant_id="t1",
            role="admin",
            roles=["admin", "editor"],
            scopes=["doc.read", "doc.write"],
        )

        mock_connector = MagicMock()
        mock_connector.get_user_context = AsyncMock(return_value=mock_user)

        mock_registry = MagicMock()
        mock_registry.get_default.return_value = mock_connector

        with patch(
            "app.services.system_context.connector_registry.get_connector_registry",
            return_value=mock_registry,
        ):
            ctx = _run(resolve_user_access_context(
                tenant_id="t1", user_id=10,
            ))

        assert ctx.user_id == "10"
        assert "admin" in ctx.role_codes
        assert "editor" in ctx.role_codes
        assert "doc.read" in ctx.permissions
        assert "doc.write" in ctx.permissions

    def test_includes_primary_role(self):
        """Primary role should be included even if not in roles list."""
        from app.schemas.system_context import UserContext

        mock_user = UserContext(
            user_id=10,
            tenant_id="t1",
            role="super_admin",
            roles=["editor"],
            scopes=[],
        )

        mock_connector = MagicMock()
        mock_connector.get_user_context = AsyncMock(return_value=mock_user)

        mock_registry = MagicMock()
        mock_registry.get_default.return_value = mock_connector

        with patch(
            "app.services.system_context.connector_registry.get_connector_registry",
            return_value=mock_registry,
        ):
            ctx = _run(resolve_user_access_context(
                tenant_id="t1", user_id=10,
            ))

        assert "super_admin" in ctx.role_codes
        assert "editor" in ctx.role_codes

    def test_fallback_when_no_connector(self):
        """Should return user_id-only context when no connector."""
        mock_registry = MagicMock()
        mock_registry.get_default.return_value = None

        with patch(
            "app.services.system_context.connector_registry.get_connector_registry",
            return_value=mock_registry,
        ):
            ctx = _run(resolve_user_access_context(
                tenant_id="t1", user_id=10,
            ))

        assert ctx.user_id == "10"
        assert ctx.role_codes == ()
        assert ctx.permissions == ()

    def test_fallback_when_connector_raises(self):
        """Should return user_id-only context on connector error."""
        mock_connector = MagicMock()
        mock_connector.get_user_context = AsyncMock(
            side_effect=RuntimeError("Connection failed"),
        )

        mock_registry = MagicMock()
        mock_registry.get_default.return_value = mock_connector

        with patch(
            "app.services.system_context.connector_registry.get_connector_registry",
            return_value=mock_registry,
        ):
            ctx = _run(resolve_user_access_context(
                tenant_id="t1", user_id=10,
            ))

        assert ctx.user_id == "10"
        assert ctx.role_codes == ()
        assert ctx.permissions == ()

    def test_fallback_when_user_not_found(self):
        """Should return user_id-only context when connector returns None."""
        mock_connector = MagicMock()
        mock_connector.get_user_context = AsyncMock(return_value=None)

        mock_registry = MagicMock()
        mock_registry.get_default.return_value = mock_connector

        with patch(
            "app.services.system_context.connector_registry.get_connector_registry",
            return_value=mock_registry,
        ):
            ctx = _run(resolve_user_access_context(
                tenant_id="t1", user_id=10,
            ))

        assert ctx.user_id == "10"
        assert ctx.role_codes == ()
        assert ctx.permissions == ()
