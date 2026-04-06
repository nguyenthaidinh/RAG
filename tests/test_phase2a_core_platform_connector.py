"""
Phase 2A — CorePlatformConnector tests.

Covers:
  1. Constructor backward compatibility (zero-arg)
  2. HTTP success with various payload shapes
  3. Tenant mismatch handling
  4. Permission fail-closed semantics
  5. Record/workflow item-level error tolerance
  6. Oversized response handling
  7. Invalid JSON handling
  8. Timeout handling
  9. build_context_bundle flag routing

Uses asyncio.run() for async tests (no pytest-asyncio dependency).
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


def _connector(**overrides):
    """Create a CorePlatformConnector with test defaults."""
    from app.services.system_context.core_platform_connector import (
        CorePlatformConnector,
    )

    defaults = {
        "base_url": "http://core.test",
        "auth_token": "test-token",
        "timeout_s": 2.0,
        "connect_timeout_s": 1.0,
        "read_timeout_s": 2.0,
        "max_response_bytes": 65536,
    }
    defaults.update(overrides)
    return CorePlatformConnector(**defaults)


def _mock_client(*, json_data=None, status_code=200,
                  content=None, side_effect=None):
    """Create mock httpx.AsyncClient for injection."""
    client = AsyncMock(spec=httpx.AsyncClient)

    if side_effect is not None:
        client.get = AsyncMock(side_effect=side_effect)
    else:
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = status_code
        resp.headers = {}

        if content is not None:
            resp.content = content
            try:
                resp.json.return_value = json.loads(content.decode())
            except (ValueError, UnicodeDecodeError):
                resp.json.side_effect = ValueError("Invalid JSON")
        elif json_data is not None:
            body = json.dumps(json_data).encode()
            resp.content = body
            resp.json.return_value = json_data
            resp.headers = {"content-length": str(len(body))}
        else:
            resp.content = b"{}"
            resp.json.return_value = {}

        resp.raise_for_status = MagicMock()
        if status_code >= 400:
            resp.raise_for_status = MagicMock(
                side_effect=httpx.HTTPStatusError(
                    "error", request=MagicMock(), response=resp,
                )
            )

        client.get = AsyncMock(return_value=resp)

    return client


# =====================================================================
# 1. CONSTRUCTOR TESTS
# =====================================================================


class TestConstructor:
    def test_default_constructor_no_args(self):
        """Registry calls CorePlatformConnector() with no args."""
        from app.services.system_context.core_platform_connector import (
            CorePlatformConnector,
        )
        c = CorePlatformConnector()
        assert c.provider_name == "core-platform"

    def test_constructor_with_overrides(self):
        c = _connector(base_url="http://custom.test", auth_token="custom")
        assert c.provider_name == "core-platform"

    def test_constructor_with_client(self):
        client = _mock_client(json_data={"user_id": "u1"})
        c = _connector(client=client)
        assert c.provider_name == "core-platform"


# =====================================================================
# 2. get_user_context
# =====================================================================


class TestGetUserContext:
    def test_success_direct_object(self):
        """Payload is a raw object (no envelope)."""
        client = _mock_client(json_data={
            "user_id": "u1",
            "tenant_id": "t1",
            "display_name": "Alice",
            "role": "admin",
            "roles": ["admin", "user"],
            "email": "alice@test.com",
        })
        c = _connector(client=client)

        result = _run(c.get_user_context(tenant_id="t1", actor_user_id="u1"))

        assert result is not None
        assert result.user_id == "u1"
        assert result.tenant_id == "t1"
        assert result.display_name == "Alice"
        assert result.role == "admin"
        assert result.roles == ["admin", "user"]

    def test_success_with_data_envelope(self):
        """Payload wrapped in {"data": {...}}."""
        client = _mock_client(json_data={
            "data": {
                "user_id": "u2",
                "tenant_id": "t1",
                "display_name": "Bob",
            }
        })
        c = _connector(client=client)

        result = _run(c.get_user_context(tenant_id="t1", actor_user_id="u2"))

        assert result is not None
        assert result.user_id == "u2"
        assert result.display_name == "Bob"

    def test_tenant_mismatch_returns_none(self):
        """Remote tenant_id differs → discard."""
        client = _mock_client(json_data={
            "user_id": "u1",
            "tenant_id": "t-other",
            "display_name": "Eve",
        })
        c = _connector(client=client)

        result = _run(c.get_user_context(tenant_id="t1", actor_user_id="u1"))
        assert result is None

    def test_http_404_returns_none(self):
        """Endpoint not deployed → fail-open None."""
        client = _mock_client(status_code=404)
        c = _connector(client=client)

        result = _run(c.get_user_context(tenant_id="t1", actor_user_id="u1"))
        assert result is None

    def test_timeout_returns_none(self):
        """Network timeout → fail-open None."""
        client = _mock_client(
            side_effect=httpx.ReadTimeout("timeout"),
        )
        c = _connector(client=client)

        result = _run(c.get_user_context(tenant_id="t1", actor_user_id="u1"))
        assert result is None

    def test_no_base_url_returns_none(self):
        """Missing base URL → fail-open None."""
        c = _connector(base_url="")

        result = _run(c.get_user_context(tenant_id="t1", actor_user_id="u1"))
        assert result is None


# =====================================================================
# 3. get_tenant_context
# =====================================================================


class TestGetTenantContext:
    def test_success(self):
        client = _mock_client(json_data={
            "tenant_id": "t1",
            "tenant_name": "Acme Corp",
            "tenant_slug": "acme",
        })
        c = _connector(client=client)

        result = _run(c.get_tenant_context(tenant_id="t1"))
        assert result is not None
        assert result.tenant_name == "Acme Corp"
        assert result.tenant_slug == "acme"

    def test_tenant_mismatch_returns_none(self):
        client = _mock_client(json_data={
            "tenant_id": "t-wrong",
            "tenant_name": "Evil Corp",
        })
        c = _connector(client=client)

        result = _run(c.get_tenant_context(tenant_id="t1"))
        assert result is None


# =====================================================================
# 4. get_permission_snapshot — FAIL-CLOSED
# =====================================================================


class TestGetPermissionSnapshot:
    def test_success(self):
        client = _mock_client(json_data={
            "tenant_id": "t1",
            "decisions": [
                {
                    "resource_type": "document",
                    "action": "read",
                    "allowed": True,
                },
                {
                    "resource_type": "user",
                    "action": "manage",
                    "allowed": False,
                    "reason": "not admin",
                },
            ],
        })
        c = _connector(client=client)

        result = _run(c.get_permission_snapshot(
            tenant_id="t1", actor_user_id="u1",
        ))
        assert result is not None
        assert result.tenant_id == "t1"
        assert len(result.decisions) == 2
        assert result.decisions[0].allowed is True
        assert result.decisions[1].allowed is False

    def test_http_error_returns_empty_snapshot_not_none(self):
        """HTTP error → empty snapshot (deny-all), NOT None."""
        client = _mock_client(status_code=500)
        c = _connector(client=client)

        result = _run(c.get_permission_snapshot(
            tenant_id="t1", actor_user_id="u1",
        ))
        assert result is not None  # Never None for permissions
        assert result.tenant_id == "t1"
        assert result.actor_user_id == "u1"
        assert len(result.decisions) == 0

    def test_404_returns_empty_snapshot(self):
        """Endpoint not deployed → empty snapshot."""
        client = _mock_client(status_code=404)
        c = _connector(client=client)

        result = _run(c.get_permission_snapshot(
            tenant_id="t1", actor_user_id="u1",
        ))
        assert result is not None
        assert len(result.decisions) == 0

    def test_timeout_returns_empty_snapshot(self):
        """Timeout → deny-all empty snapshot."""
        client = _mock_client(
            side_effect=httpx.ReadTimeout("timeout"),
        )
        c = _connector(client=client)

        result = _run(c.get_permission_snapshot(
            tenant_id="t1", actor_user_id="u1",
        ))
        assert result is not None
        assert len(result.decisions) == 0


# =====================================================================
# 5. get_record_summaries
# =====================================================================


class TestGetRecordSummaries:
    def test_success_items_envelope(self):
        client = _mock_client(json_data={
            "items": [
                {
                    "record_type": "request",
                    "record_id": "R-1",
                    "tenant_id": "t1",
                    "title": "Request One",
                    "status": "pending",
                },
                {
                    "record_type": "task",
                    "record_id": "T-1",
                    "tenant_id": "t1",
                    "title": "Task One",
                },
            ]
        })
        c = _connector(client=client)

        result = _run(c.get_record_summaries(
            tenant_id="t1", actor_user_id="u1",
        ))
        assert len(result) == 2
        assert result[0].record_type == "request"
        assert result[1].record_type == "task"

    def test_skip_invalid_keep_valid(self):
        """Invalid items are skipped, valid ones are kept."""
        client = _mock_client(json_data={
            "items": [
                {
                    "record_type": "request",
                    "record_id": "R-1",
                    "tenant_id": "t1",
                    "title": "Valid",
                },
                "not-a-dict",  # invalid
                42,            # invalid
                {
                    "record_type": "task",
                    "record_id": "T-2",
                    "tenant_id": "t1",
                    "title": "Also Valid",
                },
            ]
        })
        c = _connector(client=client)

        result = _run(c.get_record_summaries(tenant_id="t1"))
        assert len(result) == 2
        assert result[0].title == "Valid"
        assert result[1].title == "Also Valid"

    def test_tenant_mismatch_items_filtered(self):
        """Items with wrong tenant_id are filtered out."""
        client = _mock_client(json_data={
            "items": [
                {
                    "record_type": "request",
                    "record_id": "R-1",
                    "tenant_id": "t1",
                    "title": "Mine",
                },
                {
                    "record_type": "request",
                    "record_id": "R-2",
                    "tenant_id": "t-other",
                    "title": "Not Mine",
                },
                {
                    "record_type": "request",
                    "record_id": "R-3",
                    # no tenant_id → kept
                    "title": "No Tenant",
                },
            ]
        })
        c = _connector(client=client)

        result = _run(c.get_record_summaries(tenant_id="t1"))
        assert len(result) == 2
        assert result[0].title == "Mine"
        assert result[1].title == "No Tenant"


# =====================================================================
# 6. OVERSIZED RESPONSE
# =====================================================================


class TestOversizedResponse:
    def test_oversized_user_returns_none(self):
        """Response exceeds max_response_bytes → fail-open None."""
        huge_body = b'{"user_id": "u1"}' + b" " * 70000
        client = _mock_client(content=huge_body)
        c = _connector(max_response_bytes=1000, client=client)

        result = _run(c.get_user_context(tenant_id="t1", actor_user_id="u1"))
        assert result is None

    def test_oversized_records_returns_empty(self):
        """Oversized → fail-open empty list for records."""
        huge_body = b'{"items": []}' + b" " * 70000
        client = _mock_client(content=huge_body)
        c = _connector(max_response_bytes=1000, client=client)

        result = _run(c.get_record_summaries(tenant_id="t1"))
        assert result == []


# =====================================================================
# 7. INVALID JSON
# =====================================================================


class TestInvalidJSON:
    def test_invalid_json_user_returns_none(self):
        """Garbled JSON → fail-open None."""
        client = _mock_client(content=b"not json at all")
        # Make json() raise
        client.get.return_value.json.side_effect = ValueError("bad json")
        c = _connector(client=client)

        result = _run(c.get_user_context(tenant_id="t1", actor_user_id="u1"))
        assert result is None

    def test_invalid_json_permissions_returns_empty_snapshot(self):
        """Garbled JSON → fail-closed empty snapshot for permissions."""
        client = _mock_client(content=b"not json")
        client.get.return_value.json.side_effect = ValueError("bad json")
        c = _connector(client=client)

        result = _run(c.get_permission_snapshot(
            tenant_id="t1", actor_user_id="u1",
        ))
        assert result is not None
        assert len(result.decisions) == 0


# =====================================================================
# 8. TIMEOUT
# =====================================================================


class TestTimeout:
    def test_timeout_user(self):
        client = _mock_client(side_effect=httpx.ReadTimeout("timeout"))
        c = _connector(client=client)

        result = _run(c.get_user_context(tenant_id="t1", actor_user_id="u1"))
        assert result is None

    def test_timeout_tenant(self):
        client = _mock_client(side_effect=httpx.ConnectTimeout("timeout"))
        c = _connector(client=client)

        result = _run(c.get_tenant_context(tenant_id="t1"))
        assert result is None

    def test_timeout_stats(self):
        client = _mock_client(side_effect=httpx.ReadTimeout("timeout"))
        c = _connector(client=client)

        result = _run(c.get_tenant_stats(tenant_id="t1"))
        assert result is None

    def test_timeout_workflows(self):
        client = _mock_client(side_effect=httpx.ReadTimeout("timeout"))
        c = _connector(client=client)

        result = _run(c.get_workflow_summaries(tenant_id="t1"))
        assert result == []


# =====================================================================
# 9. build_context_bundle
# =====================================================================


class TestBuildContextBundle:
    def test_minimal_flags(self):
        """Only user + tenant."""
        client = _mock_client(json_data={
            "user_id": "u1",
            "tenant_id": "t1",
            "display_name": "Alice",
        })
        c = _connector(client=client)

        bundle = _run(c.build_context_bundle(
            tenant_id="t1",
            actor_user_id="u1",
            include_user=True,
            include_tenant=False,
            include_permissions=False,
        ))
        assert bundle.source == "core-platform"
        assert bundle.user is not None
        assert bundle.tenant is None
        assert bundle.permissions is None

    def test_all_flags(self):
        """All flags = True should call all methods."""
        client = _mock_client(json_data={
            "user_id": "u1",
            "tenant_id": "t1",
            "display_name": "Test",
        })
        c = _connector(client=client)

        bundle = _run(c.build_context_bundle(
            tenant_id="t1",
            actor_user_id="u1",
            include_user=True,
            include_tenant=True,
            include_permissions=True,
            include_stats=True,
            include_records=True,
            include_workflows=True,
        ))
        assert bundle.source == "core-platform"
        # At minimum, user should be populated from the mock
        assert bundle.user is not None


# =====================================================================
# 10. get_workflow_summaries
# =====================================================================


class TestGetWorkflowSummaries:
    def test_success(self):
        client = _mock_client(json_data={
            "results": [
                {
                    "workflow_type": "approval",
                    "tenant_id": "t1",
                    "total": 100,
                    "pending_count": 20,
                    "completed_count": 80,
                    "by_status": {"pending": 20, "approved": 80},
                },
            ]
        })
        c = _connector(client=client)

        result = _run(c.get_workflow_summaries(tenant_id="t1"))
        assert len(result) == 1
        assert result[0].workflow_type == "approval"
        assert result[0].total == 100

    def test_tenant_mismatch_filtered(self):
        client = _mock_client(json_data={
            "results": [
                {
                    "workflow_type": "approval",
                    "tenant_id": "t-other",
                    "total": 50,
                },
            ]
        })
        c = _connector(client=client)

        result = _run(c.get_workflow_summaries(tenant_id="t1"))
        assert result == []


# =====================================================================
# 11. get_tenant_stats
# =====================================================================


class TestGetTenantStats:
    def test_success(self):
        client = _mock_client(json_data={
            "data": {
                "tenant_id": "t1",
                "metrics": [
                    {"key": "users", "value": 42, "label": "Total Users"},
                    {"key": "docs", "value": 100},
                ],
                "period": "monthly",
            }
        })
        c = _connector(client=client)

        result = _run(c.get_tenant_stats(tenant_id="t1"))
        assert result is not None
        assert len(result.metrics) == 2
        assert result.metrics[0].key == "users"
        assert result.metrics[0].value == 42
        assert result.period == "monthly"

    def test_tenant_mismatch(self):
        client = _mock_client(json_data={
            "tenant_id": "t-wrong",
            "metrics": [{"key": "x", "value": 1}],
        })
        c = _connector(client=client)

        result = _run(c.get_tenant_stats(tenant_id="t1"))
        assert result is None
