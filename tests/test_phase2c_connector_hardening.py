"""
Phase 2C — Connector hardening tests.

Covers:
  A. Invalid content-length header does NOT crash
  B. Invalid JSON body degrades correctly
  C. Envelope data=None → object endpoint returns None
  D. List envelope shape wrong → returns []
  E. roles/scopes wrong type → fallback to []
  F. attributes/by_status/metadata wrong type → fallback to {}
  G. Bad items in list → skip item, not fail list
  H. _safe_str coercion
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from app.services.system_context.core_platform_connector import CorePlatformConnector


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _mock_response(
    *,
    status_code: int = 200,
    body: bytes | None = None,
    json_data: dict | list | None = None,
    headers: dict | None = None,
):
    """Build a mock httpx.Response."""
    if headers is None:
        headers = {}
    if body is None and json_data is not None:
        body = json.dumps(json_data).encode()
    if body is None:
        body = b""

    resp = httpx.Response(
        status_code=status_code,
        content=body,
        headers=headers,
        request=httpx.Request("GET", "http://test/api/internal/context/user"),
    )
    return resp


def _make_connector(mock_response):
    """Create connector with a mock client that returns the given response."""
    client = AsyncMock()
    client.get = AsyncMock(return_value=mock_response)
    return CorePlatformConnector(
        base_url="http://test",
        auth_token="tok",
        client=client,
    )


# =====================================================================
# A. INVALID CONTENT-LENGTH
# =====================================================================


class TestInvalidContentLength:
    def test_non_integer_content_length_does_not_crash(self):
        """content-length='abc' → skip check, continue to body."""
        resp = _mock_response(
            json_data={"user_id": "u1", "display_name": "Alice"},
            headers={"content-length": "abc"},
        )
        conn = _make_connector(resp)
        result = _run(conn.get_user_context(tenant_id="t1", actor_user_id="u1"))
        assert result is not None
        assert result.display_name == "Alice"

    def test_empty_content_length_does_not_crash(self):
        """content-length='' → skip check."""
        resp = _mock_response(
            json_data={"user_id": "u1", "display_name": "Bob"},
            headers={"content-length": ""},
        )
        conn = _make_connector(resp)
        result = _run(conn.get_user_context(tenant_id="t1", actor_user_id="u1"))
        assert result is not None


# =====================================================================
# B. INVALID JSON BODY
# =====================================================================


class TestInvalidJsonBody:
    def test_user_context_invalid_json_returns_none(self):
        """Non-JSON body → fail-open → None."""
        resp = _mock_response(
            body=b"<html>error</html>",
            headers={"content-type": "text/html"},
        )
        conn = _make_connector(resp)
        result = _run(conn.get_user_context(tenant_id="t1", actor_user_id="u1"))
        assert result is None

    def test_records_invalid_json_returns_empty(self):
        """Non-JSON body on list endpoint → fail-open → []."""
        resp = _mock_response(body=b"not json")
        conn = _make_connector(resp)
        result = _run(conn.get_record_summaries(tenant_id="t1"))
        assert result == []

    def test_permissions_invalid_json_returns_empty_snapshot(self):
        """Non-JSON body → fail-closed → empty snapshot."""
        resp = _mock_response(body=b"broken")
        conn = _make_connector(resp)
        result = _run(conn.get_permission_snapshot(
            tenant_id="t1", actor_user_id="u1",
        ))
        assert result.decisions == []


# =====================================================================
# C. ENVELOPE data=None
# =====================================================================


class TestEnvelopeDataNull:
    def test_object_endpoint_data_null_returns_none(self):
        """Object endpoint with {data: null} → None."""
        resp = _mock_response(json_data={"data": None})
        conn = _make_connector(resp)
        result = _run(conn.get_user_context(tenant_id="t1", actor_user_id="u1"))
        assert result is None

    def test_list_endpoint_data_null_returns_empty_list(self):
        """List endpoint with {data: null} → []."""
        resp = _mock_response(json_data={"data": None})
        conn = _make_connector(resp)
        result = _run(conn.get_record_summaries(tenant_id="t1"))
        assert result == []

    def test_items_null_returns_empty(self):
        """{items: null} → []."""
        resp = _mock_response(json_data={"items": None})
        conn = _make_connector(resp)
        result = _run(conn.get_workflow_summaries(tenant_id="t1"))
        assert result == []


# =====================================================================
# D. LIST ENDPOINT WRONG SHAPE
# =====================================================================


class TestListEndpointWrongShape:
    def test_data_is_string_returns_empty(self):
        """{data: 'oops'} on list endpoint → []."""
        resp = _mock_response(json_data={"data": "not a list"})
        conn = _make_connector(resp)
        result = _run(conn.get_record_summaries(tenant_id="t1"))
        assert result == []

    def test_items_is_dict_returns_empty(self):
        """{items: {}} on list endpoint → []."""
        resp = _mock_response(json_data={"items": {"wrong": "shape"}})
        conn = _make_connector(resp)
        result = _run(conn.get_workflow_summaries(tenant_id="t1"))
        assert result == []


# =====================================================================
# E. ROLES/SCOPES WRONG TYPE → []
# =====================================================================


class TestRolesScopesCoercion:
    def test_roles_is_string_falls_back_to_empty(self):
        """roles='admin' instead of list → []."""
        resp = _mock_response(json_data={
            "user_id": "u1", "display_name": "Alice",
            "roles": "admin",
            "scopes": ["read"],
        })
        conn = _make_connector(resp)
        result = _run(conn.get_user_context(tenant_id="t1", actor_user_id="u1"))
        assert result is not None
        assert result.roles == []
        assert result.scopes == ["read"]

    def test_scopes_is_int_falls_back_to_empty(self):
        """scopes=42 → []."""
        resp = _mock_response(json_data={
            "user_id": "u1", "display_name": "Bob",
            "scopes": 42,
        })
        conn = _make_connector(resp)
        result = _run(conn.get_user_context(tenant_id="t1", actor_user_id="u1"))
        assert result is not None
        assert result.scopes == []

    def test_role_is_int_coerced_to_string(self):
        """role=123 → '123'."""
        resp = _mock_response(json_data={
            "user_id": "u1", "display_name": "Charlie",
            "role": 123,
        })
        conn = _make_connector(resp)
        result = _run(conn.get_user_context(tenant_id="t1", actor_user_id="u1"))
        assert result is not None
        assert result.role == "123"


# =====================================================================
# F. ATTRIBUTES/METADATA/BY_STATUS WRONG TYPE → {}
# =====================================================================


class TestDictFieldCoercion:
    def test_user_attributes_is_string_falls_back(self):
        """attributes='bad' → {}."""
        resp = _mock_response(json_data={
            "user_id": "u1",
            "attributes": "not a dict",
        })
        conn = _make_connector(resp)
        result = _run(conn.get_user_context(tenant_id="t1", actor_user_id="u1"))
        assert result is not None
        assert result.attributes == {}

    def test_tenant_attributes_is_list_falls_back(self):
        resp = _mock_response(json_data={
            "tenant_name": "Acme",
            "attributes": [1, 2],
        })
        conn = _make_connector(resp)
        result = _run(conn.get_tenant_context(tenant_id="t1"))
        assert result is not None
        assert result.attributes == {}

    def test_record_metadata_is_list_falls_back(self):
        resp = _mock_response(json_data=[
            {"record_type": "task", "record_id": "R1", "metadata": [1]},
        ])
        conn = _make_connector(resp)
        result = _run(conn.get_record_summaries(tenant_id="t1"))
        assert len(result) == 1
        assert result[0].metadata == {}

    def test_workflow_by_status_is_string_falls_back(self):
        resp = _mock_response(json_data=[
            {"workflow_type": "approval", "by_status": "bad"},
        ])
        conn = _make_connector(resp)
        result = _run(conn.get_workflow_summaries(tenant_id="t1"))
        assert len(result) == 1
        assert result[0].by_status == {}

    def test_permission_field_masking_is_string_falls_back(self):
        resp = _mock_response(json_data={
            "decisions": [
                {"resource_type": "doc", "action": "read",
                 "allowed": True, "field_masking": "bad"},
            ],
        })
        conn = _make_connector(resp)
        result = _run(conn.get_permission_snapshot(
            tenant_id="t1", actor_user_id="u1",
        ))
        assert len(result.decisions) == 1
        assert result.decisions[0].field_masking == {}


# =====================================================================
# G. BAD ITEMS IN LIST → SKIP ITEM
# =====================================================================


class TestBadItemSkipping:
    def test_record_bad_item_skipped(self):
        """One good + one bad item → only good item returned."""
        resp = _mock_response(json_data=[
            {"record_type": "task", "record_id": "R1", "title": "Good"},
            42,  # bad item
            {"record_type": "note", "record_id": "R2", "title": "Also Good"},
        ])
        conn = _make_connector(resp)
        result = _run(conn.get_record_summaries(tenant_id="t1"))
        assert len(result) == 2
        assert result[0].title == "Good"
        assert result[1].title == "Also Good"

    def test_workflow_bad_item_skipped(self):
        resp = _mock_response(json_data=[
            "not a dict",
            {"workflow_type": "approval", "total": 10},
        ])
        conn = _make_connector(resp)
        result = _run(conn.get_workflow_summaries(tenant_id="t1"))
        assert len(result) == 1
        assert result[0].workflow_type == "approval"

    def test_permission_bad_decision_skipped(self):
        resp = _mock_response(json_data={
            "decisions": [
                {"resource_type": "doc", "action": "read", "allowed": True},
                "bad",
                {"resource_type": "user", "action": "manage", "allowed": False},
            ],
        })
        conn = _make_connector(resp)
        result = _run(conn.get_permission_snapshot(
            tenant_id="t1", actor_user_id="u1",
        ))
        assert len(result.decisions) == 2


# =====================================================================
# H. _safe_str COERCION
# =====================================================================


class TestSafeStr:
    def test_none_returns_none(self):
        from app.services.system_context.core_platform_connector import _safe_str
        assert _safe_str(None) is None

    def test_string_passthrough(self):
        from app.services.system_context.core_platform_connector import _safe_str
        assert _safe_str("hello") == "hello"

    def test_int_coerced(self):
        from app.services.system_context.core_platform_connector import _safe_str
        assert _safe_str(42) == "42"

    def test_bool_coerced(self):
        from app.services.system_context.core_platform_connector import _safe_str
        assert _safe_str(True) == "True"

    def test_float_coerced(self):
        from app.services.system_context.core_platform_connector import _safe_str
        assert _safe_str(3.14) == "3.14"
