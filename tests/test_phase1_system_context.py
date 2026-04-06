"""
Phase 1 — System Context Foundation tests.

Covers:
  1. Schema validation (defaults, serialization, properties)
  2. Connector tests (registry, mock, placeholder)
  3. Question classifier tests (all categories)
  4. Context builder tests (full, partial, failure handling)
  5. Ingest-reference hardening tests (allowlist validation)
  6. Orchestration foundation tests

Uses asyncio.run() for async tests (no pytest-asyncio dependency).
"""
from __future__ import annotations

import asyncio
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


# =====================================================================
# 1. SCHEMA TESTS
# =====================================================================


class TestUserContext:
    def test_defaults(self):
        from app.schemas.system_context import UserContext

        ctx = UserContext(user_id="u1", tenant_id="t1")
        assert ctx.user_id == "u1"
        assert ctx.tenant_id == "t1"
        assert ctx.email is None
        assert ctx.display_name is None
        assert ctx.role is None
        assert ctx.roles == []
        assert ctx.scopes == []
        assert ctx.attributes == {}
        assert ctx.is_active is True

    def test_full_fields(self):
        from app.schemas.system_context import UserContext

        ctx = UserContext(
            user_id=42,
            email="test@example.com",
            display_name="Test User",
            tenant_id="t1",
            role="admin",
            roles=["admin", "user"],
            scopes=["read", "write"],
            attributes={"team": "engineering"},
            is_active=False,
        )
        assert ctx.user_id == 42
        assert ctx.email == "test@example.com"
        assert ctx.is_active is False

    def test_serialization(self):
        from app.schemas.system_context import UserContext

        ctx = UserContext(user_id="u1", tenant_id="t1")
        d = ctx.model_dump()
        assert d["user_id"] == "u1"
        assert d["tenant_id"] == "t1"
        assert isinstance(d["roles"], list)


class TestTenantContext:
    def test_defaults(self):
        from app.schemas.system_context import TenantContext

        ctx = TenantContext(tenant_id="t1")
        assert ctx.tenant_id == "t1"
        assert ctx.tenant_name is None
        assert ctx.tenant_slug is None
        assert ctx.attributes == {}
        assert isinstance(ctx.fetched_at, datetime)

    def test_serialization(self):
        from app.schemas.system_context import TenantContext

        ctx = TenantContext(
            tenant_id="t1",
            tenant_name="Test Tenant",
            tenant_slug="test-tenant",
        )
        d = ctx.model_dump(mode="json")
        assert d["tenant_id"] == "t1"
        assert d["tenant_name"] == "Test Tenant"


class TestPermissionDecision:
    def test_create(self):
        from app.schemas.system_context import PermissionDecision

        pd = PermissionDecision(
            resource_type="document",
            action="read",
            allowed=True,
        )
        assert pd.resource_type == "document"
        assert pd.allowed is True
        assert pd.field_masking == {}

    def test_with_masking(self):
        from app.schemas.system_context import PermissionDecision

        pd = PermissionDecision(
            resource_type="user",
            action="view",
            allowed=True,
            field_masking={"email": "masked", "phone": "hidden"},
        )
        assert pd.field_masking["email"] == "masked"


class TestPermissionSnapshot:
    def test_empty(self):
        from app.schemas.system_context import PermissionSnapshot

        snap = PermissionSnapshot(tenant_id="t1", actor_user_id="u1")
        assert snap.decisions == []
        assert snap.actor_user_id == "u1"

    def test_with_decisions(self):
        from app.schemas.system_context import PermissionDecision, PermissionSnapshot

        snap = PermissionSnapshot(
            tenant_id="t1",
            actor_user_id=42,
            decisions=[
                PermissionDecision(
                    resource_type="doc", action="read", allowed=True,
                ),
            ],
        )
        assert len(snap.decisions) == 1
        assert snap.decisions[0].allowed is True


class TestMetricValue:
    def test_int_value(self):
        from app.schemas.system_context import MetricValue

        mv = MetricValue(key="total", value=42)
        assert mv.value == 42
        assert mv.label is None

    def test_string_value(self):
        from app.schemas.system_context import MetricValue

        mv = MetricValue(key="version", value="1.0", label="Version")
        assert mv.value == "1.0"

    def test_bool_value(self):
        from app.schemas.system_context import MetricValue

        mv = MetricValue(key="active", value=True)
        assert mv.value is True


class TestTenantStats:
    def test_defaults(self):
        from app.schemas.system_context import TenantStats

        stats = TenantStats(tenant_id="t1")
        assert stats.metrics == []
        assert stats.period is None

    def test_with_metrics(self):
        from app.schemas.system_context import MetricValue, TenantStats

        stats = TenantStats(
            tenant_id="t1",
            metrics=[MetricValue(key="users", value=10)],
            period="monthly",
        )
        assert len(stats.metrics) == 1


class TestRecordSummary:
    def test_minimal(self):
        from app.schemas.system_context import RecordSummary

        rs = RecordSummary(record_type="request", record_id="R-1", tenant_id="t1")
        assert rs.record_type == "request"
        assert rs.title is None
        assert rs.metadata == {}

    def test_full(self):
        from app.schemas.system_context import RecordSummary

        rs = RecordSummary(
            record_type="request",
            record_id="R-1",
            title="Test Request",
            status="pending",
            owner_id="u1",
            tenant_id="t1",
            summary="A test request",
            metadata={"priority": "high"},
        )
        assert rs.title == "Test Request"
        assert rs.metadata["priority"] == "high"


class TestWorkflowSummary:
    def test_minimal(self):
        from app.schemas.system_context import WorkflowSummary

        ws = WorkflowSummary(workflow_type="approval", tenant_id="t1")
        assert ws.total is None
        assert ws.by_status == {}

    def test_full(self):
        from app.schemas.system_context import WorkflowSummary

        ws = WorkflowSummary(
            workflow_type="approval",
            tenant_id="t1",
            total=100,
            by_status={"pending": 20, "approved": 80},
            pending_count=20,
            completed_count=80,
        )
        assert ws.total == 100
        assert ws.by_status["pending"] == 20


class TestSystemContextBundle:
    def test_empty_bundle(self):
        from app.schemas.system_context import SystemContextBundle

        b = SystemContextBundle()
        assert b.user is None
        assert b.tenant is None
        assert b.permissions is None
        assert b.tenant_stats is None
        assert b.records == []
        assert b.workflows == []
        assert b.source == "unknown"
        assert not b.has_user
        assert not b.has_permissions
        assert not b.has_stats

    def test_properties(self):
        from app.schemas.system_context import (
            MetricValue,
            PermissionDecision,
            PermissionSnapshot,
            SystemContextBundle,
            TenantStats,
            UserContext,
        )

        b = SystemContextBundle(
            user=UserContext(user_id="u1", tenant_id="t1"),
            permissions=PermissionSnapshot(
                tenant_id="t1",
                actor_user_id="u1",
                decisions=[
                    PermissionDecision(
                        resource_type="doc", action="read", allowed=True,
                    ),
                ],
            ),
            tenant_stats=TenantStats(
                tenant_id="t1",
                metrics=[MetricValue(key="x", value=1)],
            ),
            source="mock",
        )
        assert b.has_user is True
        assert b.has_permissions is True
        assert b.has_stats is True

    def test_serialization(self):
        from app.schemas.system_context import SystemContextBundle

        b = SystemContextBundle(source="test")
        d = b.model_dump(mode="json")
        assert d["source"] == "test"
        assert isinstance(d["fetched_at"], str)


# =====================================================================
# 2. CONNECTOR TESTS
# =====================================================================


class TestConnectorRegistry:
    def test_register_and_get(self):
        from app.services.system_context.connector_registry import ConnectorRegistry

        registry = ConnectorRegistry()
        mock = MagicMock()
        mock.provider_name = "test"
        registry.register("test", mock)

        assert registry.get("test") is mock
        assert "test" in registry
        assert "nonexistent" not in registry

    def test_default_connector(self):
        from app.services.system_context.connector_registry import ConnectorRegistry

        registry = ConnectorRegistry()
        mock1 = MagicMock()
        mock1.provider_name = "first"
        mock2 = MagicMock()
        mock2.provider_name = "second"

        registry.register("first", mock1)
        registry.register("second", mock2, default=True)

        assert registry.get_default() is mock2
        assert registry.default_name == "second"

    def test_first_registered_is_default(self):
        from app.services.system_context.connector_registry import ConnectorRegistry

        registry = ConnectorRegistry()
        mock = MagicMock()
        mock.provider_name = "only"

        registry.register("only", mock)
        assert registry.get_default() is mock

    def test_list_providers(self):
        from app.services.system_context.connector_registry import ConnectorRegistry

        registry = ConnectorRegistry()
        mock = MagicMock()
        registry.register("a", mock)
        registry.register("b", mock)

        assert sorted(registry.list_providers()) == ["a", "b"]

    def test_empty_registry(self):
        from app.services.system_context.connector_registry import ConnectorRegistry

        registry = ConnectorRegistry()
        assert registry.get_default() is None
        assert registry.get("anything") is None
        assert registry.list_providers() == []

    def test_global_registry_has_mock(self):
        from app.services.system_context.connector_registry import get_connector_registry

        registry = get_connector_registry()
        assert "mock" in registry
        mock_connector = registry.get("mock")
        assert mock_connector is not None
        assert mock_connector.provider_name == "mock"


class TestMockConnector:
    def _connector(self):
        from app.services.system_context.mock_connector import MockSystemConnector

        return MockSystemConnector()

    def test_provider_name(self):
        assert self._connector().provider_name == "mock"

    def test_get_user_context(self):
        c = self._connector()
        result = _run(c.get_user_context(tenant_id="t1", actor_user_id=42))
        assert result.user_id == 42
        assert result.tenant_id == "t1"
        assert "t1" in result.email
        assert result.is_active is True

    def test_get_tenant_context(self):
        c = self._connector()
        result = _run(c.get_tenant_context(tenant_id="t1"))
        assert result.tenant_id == "t1"
        assert result.tenant_name is not None

    def test_get_permission_snapshot(self):
        c = self._connector()
        result = _run(c.get_permission_snapshot(
            tenant_id="t1", actor_user_id=42,
        ))
        assert result.tenant_id == "t1"
        assert len(result.decisions) > 0
        assert all(d.allowed for d in result.decisions)

    def test_get_permission_snapshot_filtered(self):
        c = self._connector()
        result = _run(c.get_permission_snapshot(
            tenant_id="t1", actor_user_id=42,
            resource_types=["document"],
        ))
        assert len(result.decisions) == 1
        assert result.decisions[0].resource_type == "document"

    def test_get_tenant_stats(self):
        c = self._connector()
        result = _run(c.get_tenant_stats(tenant_id="t1"))
        assert result.tenant_id == "t1"
        assert len(result.metrics) > 0

    def test_get_record_summaries(self):
        c = self._connector()
        result = _run(c.get_record_summaries(
            tenant_id="t1", actor_user_id=42,
        ))
        assert len(result) > 0
        assert result[0].tenant_id == "t1"

    def test_get_workflow_summaries(self):
        c = self._connector()
        result = _run(c.get_workflow_summaries(tenant_id="t1"))
        assert len(result) > 0
        assert result[0].workflow_type == "approval"

    def test_build_context_bundle_full(self):
        c = self._connector()
        bundle = _run(c.build_context_bundle(
            tenant_id="t1",
            actor_user_id=42,
            include_user=True,
            include_tenant=True,
            include_permissions=True,
            include_stats=True,
            include_records=True,
            include_workflows=True,
        ))
        assert bundle.source == "mock"
        assert bundle.has_user
        assert bundle.tenant is not None
        assert bundle.has_permissions
        assert bundle.has_stats
        assert len(bundle.records) > 0
        assert len(bundle.workflows) > 0

    def test_build_context_bundle_minimal(self):
        c = self._connector()
        bundle = _run(c.build_context_bundle(
            tenant_id="t1",
            actor_user_id=42,
            include_user=True,
            include_tenant=False,
            include_permissions=False,
            include_stats=False,
            include_records=False,
            include_workflows=False,
        ))
        assert bundle.has_user
        assert bundle.tenant is None
        assert not bundle.has_permissions
        assert bundle.records == []

    def test_tenant_isolation(self):
        c = self._connector()
        ctx1 = _run(c.get_user_context(tenant_id="t1", actor_user_id=1))
        ctx2 = _run(c.get_user_context(tenant_id="t2", actor_user_id=1))
        assert ctx1.tenant_id == "t1"
        assert ctx2.tenant_id == "t2"


class TestCorePlatformConnector:
    def _connector(self):
        from app.services.system_context.core_platform_connector import (
            CorePlatformConnector,
        )

        return CorePlatformConnector()

    def test_provider_name(self):
        assert self._connector().provider_name == "core-platform"

    def test_returns_none_for_user(self):
        c = self._connector()
        result = _run(c.get_user_context(tenant_id="t1", actor_user_id=1))
        assert result is None

    def test_returns_none_for_tenant(self):
        c = self._connector()
        result = _run(c.get_tenant_context(tenant_id="t1"))
        assert result is None

    def test_returns_empty_snapshot_for_permissions(self):
        """Phase 2A: real connector returns empty snapshot (fail-closed), not None."""
        c = self._connector()
        result = _run(c.get_permission_snapshot(
            tenant_id="t1", actor_user_id=1,
        ))
        # Fail-closed: empty snapshot (no base URL → no data → deny-all)
        assert result is not None
        assert result.tenant_id == "t1"
        assert len(result.decisions) == 0

    def test_returns_none_for_stats(self):
        c = self._connector()
        result = _run(c.get_tenant_stats(tenant_id="t1"))
        assert result is None

    def test_returns_empty_for_records(self):
        c = self._connector()
        result = _run(c.get_record_summaries(tenant_id="t1"))
        assert result == []

    def test_returns_empty_for_workflows(self):
        c = self._connector()
        result = _run(c.get_workflow_summaries(tenant_id="t1"))
        assert result == []

    def test_build_bundle_returns_empty(self):
        """Phase 2A: source is 'core-platform' (no longer placeholder)."""
        c = self._connector()
        bundle = _run(c.build_context_bundle(
            tenant_id="t1", actor_user_id=1,
        ))
        assert bundle.source == "core-platform"
        assert not bundle.has_user


# =====================================================================
# 3. QUESTION CLASSIFIER TESTS
# =====================================================================


class TestQuestionClassifier:
    def _classifier(self):
        from app.services.orchestration.question_classifier import QuestionClassifier

        return QuestionClassifier()

    def test_knowledge_question_vi(self):
        from app.services.orchestration.question_classifier import QuestionCategory

        result = self._classifier().classify("Tài liệu quy định về nghỉ phép là gì?")
        assert result.category == QuestionCategory.KNOWLEDGE
        assert result.confidence > 0.5
        assert len(result.matched_signals) > 0

    def test_knowledge_question_en(self):
        from app.services.orchestration.question_classifier import QuestionCategory

        result = self._classifier().classify("What is the policy for annual leave?")
        assert result.category == QuestionCategory.KNOWLEDGE

    def test_system_question_vi(self):
        from app.services.orchestration.question_classifier import QuestionCategory

        result = self._classifier().classify("Tổng số hồ sơ đang chờ duyệt là bao nhiêu?")
        assert result.category == QuestionCategory.SYSTEM
        assert result.confidence > 0.5

    def test_system_question_en(self):
        from app.services.orchestration.question_classifier import QuestionCategory

        result = self._classifier().classify("How many pending records are there?")
        assert result.category == QuestionCategory.SYSTEM

    def test_access_question_vi(self):
        from app.services.orchestration.question_classifier import QuestionCategory

        result = self._classifier().classify("Tôi có quyền xem được gì?")
        assert result.category == QuestionCategory.ACCESS
        assert result.confidence > 0.5

    def test_access_question_en(self):
        from app.services.orchestration.question_classifier import QuestionCategory

        result = self._classifier().classify("What are my permissions?")
        assert result.category == QuestionCategory.ACCESS

    def test_mixed_question(self):
        from app.services.orchestration.question_classifier import QuestionCategory

        result = self._classifier().classify(
            "Tổng số hồ sơ theo quy định mới nhất là bao nhiêu?"
        )
        assert result.category == QuestionCategory.MIXED
        assert result.secondary_category is not None

    def test_unknown_question(self):
        from app.services.orchestration.question_classifier import QuestionCategory

        result = self._classifier().classify("Xin chào, bạn khỏe không?")
        assert result.category == QuestionCategory.UNKNOWN
        assert result.confidence <= 0.5

    def test_empty_question(self):
        from app.services.orchestration.question_classifier import QuestionCategory

        result = self._classifier().classify("")
        assert result.category == QuestionCategory.UNKNOWN
        assert result.confidence == 0.0

    def test_whitespace_question(self):
        from app.services.orchestration.question_classifier import QuestionCategory

        result = self._classifier().classify("   ")
        assert result.category == QuestionCategory.UNKNOWN

    def test_telemetry_dict(self):
        result = self._classifier().classify("What is the policy?")
        td = result.telemetry_dict()
        assert "category" in td
        assert "confidence" in td
        assert "signal_count" in td

    def test_classifier_is_deterministic(self):
        c = self._classifier()
        q = "Tổng số nhân viên là bao nhiêu?"
        r1 = c.classify(q)
        r2 = c.classify(q)
        assert r1.category == r2.category
        assert r1.confidence == r2.confidence
        assert r1.matched_signals == r2.matched_signals


# =====================================================================
# 4. CONTEXT BUILDER TESTS
# =====================================================================


class TestContextBuilder:
    def _make_builder(self):
        from app.services.system_context.mock_connector import MockSystemConnector
        from app.services.system_context.context_builder import SystemContextBuilder

        return SystemContextBuilder(connector=MockSystemConnector())

    def test_build_full_bundle(self):
        from app.services.system_context.context_builder import ContextBuildFlags

        builder = self._make_builder()
        bundle = _run(builder.build(
            tenant_id="t1",
            actor_user_id=42,
            flags=ContextBuildFlags(
                include_user=True,
                include_tenant=True,
                include_permissions=True,
                include_stats=True,
                include_records=True,
                include_workflows=True,
            ),
        ))
        assert bundle.has_user
        assert bundle.tenant is not None
        assert bundle.has_permissions
        assert bundle.has_stats
        assert len(bundle.records) > 0
        assert len(bundle.workflows) > 0
        assert bundle.source == "mock"

    def test_build_partial_bundle(self):
        from app.services.system_context.context_builder import ContextBuildFlags

        builder = self._make_builder()
        bundle = _run(builder.build(
            tenant_id="t1",
            actor_user_id=42,
            flags=ContextBuildFlags(
                include_user=True,
                include_tenant=False,
                include_permissions=False,
                include_stats=False,
                include_records=False,
                include_workflows=False,
            ),
        ))
        assert bundle.has_user
        assert bundle.tenant is None
        assert not bundle.has_permissions
        assert bundle.records == []

    def test_build_default_flags(self):
        builder = self._make_builder()
        bundle = _run(builder.build(
            tenant_id="t1",
            actor_user_id=42,
        ))
        # Default flags: user + tenant only
        assert bundle.has_user
        assert bundle.tenant is not None
        assert not bundle.has_permissions

    def test_tenant_propagation(self):
        builder = self._make_builder()
        bundle = _run(builder.build(
            tenant_id="specific-tenant",
            actor_user_id=99,
        ))
        assert bundle.user.tenant_id == "specific-tenant"

    def test_failure_handling_stats(self):
        """Stats failure should not crash the builder (fail-open)."""
        from app.services.system_context.context_builder import (
            ContextBuildFlags,
            SystemContextBuilder,
        )

        connector = AsyncMock()
        connector.provider_name = "failing"
        connector.get_user_context = AsyncMock(return_value=None)
        connector.get_tenant_context = AsyncMock(return_value=None)
        connector.get_tenant_stats = AsyncMock(side_effect=Exception("stats fail"))
        connector.get_record_summaries = AsyncMock(return_value=[])
        connector.get_workflow_summaries = AsyncMock(return_value=[])

        builder = SystemContextBuilder(connector=connector)
        bundle = _run(builder.build(
            tenant_id="t1",
            actor_user_id=1,
            flags=ContextBuildFlags(include_stats=True),
        ))
        # Should not raise; stats just missing
        assert not bundle.has_stats

    def test_failure_handling_permissions_fail_closed(self):
        """Permission failure should result in empty snapshot (fail-closed)."""
        from app.services.system_context.context_builder import (
            ContextBuildFlags,
            SystemContextBuilder,
        )

        connector = AsyncMock()
        connector.provider_name = "failing"
        connector.get_user_context = AsyncMock(return_value=None)
        connector.get_tenant_context = AsyncMock(return_value=None)
        connector.get_permission_snapshot = AsyncMock(
            side_effect=Exception("perm fail"),
        )

        builder = SystemContextBuilder(connector=connector)
        bundle = _run(builder.build(
            tenant_id="t1",
            actor_user_id=1,
            flags=ContextBuildFlags(include_permissions=True),
        ))
        # Fail-closed: should have an empty snapshot (no decisions = deny all)
        assert bundle.permissions is not None
        assert len(bundle.permissions.decisions) == 0

    def test_provider_name(self):
        builder = self._make_builder()
        assert builder.provider_name == "mock"


# =====================================================================
# 5. INGEST-REFERENCE HARDENING TESTS
# =====================================================================


class TestRemoteFetchUrlValidation:
    def test_valid_url_no_enforcement(self):
        from app.services.remote_file_fetcher import validate_fetch_url

        result = validate_fetch_url(
            "https://files.example.com/doc.pdf",
            allowed_hosts=frozenset({"files.example.com"}),
            enforce=False,
        )
        assert result == "https://files.example.com/doc.pdf"

    def test_valid_url_with_enforcement(self):
        from app.services.remote_file_fetcher import validate_fetch_url

        result = validate_fetch_url(
            "https://files.example.com/doc.pdf",
            allowed_hosts=frozenset({"files.example.com"}),
            enforce=True,
        )
        assert result == "https://files.example.com/doc.pdf"

    def test_disallowed_host_enforced(self):
        from app.services.remote_file_fetcher import (
            RemoteFetchForbiddenHost,
            validate_fetch_url,
        )

        with pytest.raises(RemoteFetchForbiddenHost) as exc_info:
            validate_fetch_url(
                "https://evil.example.com/malware.exe",
                allowed_hosts=frozenset({"files.example.com"}),
                enforce=True,
            )
        assert exc_info.value.status_code == 403
        assert "evil.example.com" in str(exc_info.value)

    def test_disallowed_host_audit_only(self):
        """When enforce=False, disallowed hosts are logged but allowed."""
        from app.services.remote_file_fetcher import validate_fetch_url

        # Should NOT raise
        result = validate_fetch_url(
            "https://evil.example.com/doc.pdf",
            allowed_hosts=frozenset({"files.example.com"}),
            enforce=False,
        )
        assert result == "https://evil.example.com/doc.pdf"

    def test_malformed_url_bad_scheme(self):
        from app.services.remote_file_fetcher import RemoteFetchError, validate_fetch_url

        with pytest.raises(RemoteFetchError) as exc_info:
            validate_fetch_url(
                "ftp://files.example.com/doc.pdf",
                enforce=True,
            )
        assert exc_info.value.status_code == 400

    def test_no_hostname(self):
        from app.services.remote_file_fetcher import RemoteFetchError, validate_fetch_url

        with pytest.raises(RemoteFetchError):
            validate_fetch_url(
                "http:///path/to/file",
                enforce=True,
            )

    def test_empty_allowlist_skips_check(self):
        from app.services.remote_file_fetcher import validate_fetch_url

        result = validate_fetch_url(
            "https://any.host.com/doc.pdf",
            allowed_hosts=frozenset(),
            enforce=True,
        )
        assert result == "https://any.host.com/doc.pdf"

    def test_none_allowlist_skips_check(self):
        from app.services.remote_file_fetcher import validate_fetch_url

        result = validate_fetch_url(
            "https://any.host.com/doc.pdf",
            allowed_hosts=None,
            enforce=True,
        )
        assert result == "https://any.host.com/doc.pdf"

    def test_provider_in_error_context(self):
        from app.services.remote_file_fetcher import (
            RemoteFetchForbiddenHost,
            validate_fetch_url,
        )

        with pytest.raises(RemoteFetchForbiddenHost):
            validate_fetch_url(
                "https://bad.host.com/file.pdf",
                allowed_hosts=frozenset({"good.host.com"}),
                enforce=True,
                provider="file-service",
            )

    def test_case_insensitive_host(self):
        from app.services.remote_file_fetcher import validate_fetch_url

        result = validate_fetch_url(
            "https://Files.Example.COM/doc.pdf",
            allowed_hosts=frozenset({"files.example.com"}),
            enforce=True,
        )
        assert result is not None


class TestParseAllowedHosts:
    def test_empty_string(self):
        from app.services.remote_file_fetcher import _parse_allowed_hosts

        assert _parse_allowed_hosts("") == frozenset()

    def test_single_host(self):
        from app.services.remote_file_fetcher import _parse_allowed_hosts

        result = _parse_allowed_hosts("files.example.com")
        assert result == frozenset({"files.example.com"})

    def test_multiple_hosts(self):
        from app.services.remote_file_fetcher import _parse_allowed_hosts

        result = _parse_allowed_hosts("a.com, b.com , c.com")
        assert result == frozenset({"a.com", "b.com", "c.com"})

    def test_whitespace_handling(self):
        from app.services.remote_file_fetcher import _parse_allowed_hosts

        result = _parse_allowed_hosts("  host1.com  ,  host2.com  ")
        assert result == frozenset({"host1.com", "host2.com"})

    def test_case_normalized(self):
        from app.services.remote_file_fetcher import _parse_allowed_hosts

        result = _parse_allowed_hosts("HOST.COM")
        assert "host.com" in result


class TestRemoteFetchForbiddenHost:
    def test_exception_attributes(self):
        from app.services.remote_file_fetcher import RemoteFetchForbiddenHost

        exc = RemoteFetchForbiddenHost("evil.com")
        assert exc.status_code == 403
        assert "evil.com" in str(exc)


# =====================================================================
# 6. ORCHESTRATION FOUNDATION TESTS
# =====================================================================


class TestOrchestrationResult:
    def test_telemetry_dict(self):
        from app.services.orchestration.system_context_orchestrator import (
            OrchestrationResult,
        )
        from app.services.orchestration.question_classifier import (
            ClassificationResult,
            QuestionCategory,
        )
        from app.services.system_context.context_builder import ContextBuildFlags

        result = OrchestrationResult(
            category=QuestionCategory.KNOWLEDGE,
            classification=ClassificationResult(
                category=QuestionCategory.KNOWLEDGE,
                confidence=0.8,
                matched_signals=("knowledge:policy",),
            ),
            should_use_knowledge=True,
            should_use_system_context=False,
            should_use_access_context=False,
            recommended_flags=ContextBuildFlags(),
        )
        td = result.telemetry_dict()
        assert td["category"] == "knowledge"
        assert td["should_use_knowledge"] is True
        assert td["should_use_system_context"] is False


class TestSystemContextOrchestrator:
    def _make_orchestrator(self):
        from app.services.system_context.mock_connector import MockSystemConnector
        from app.services.system_context.context_builder import SystemContextBuilder
        from app.services.orchestration.system_context_orchestrator import (
            SystemContextOrchestrator,
        )

        connector = MockSystemConnector()
        builder = SystemContextBuilder(connector=connector)
        return SystemContextOrchestrator(context_builder=builder)

    def test_knowledge_question(self):
        from app.services.orchestration.question_classifier import QuestionCategory

        orch = self._make_orchestrator()
        result = _run(orch.evaluate(
            question="Quy định về nghỉ phép là gì?",
            tenant_id="t1",
            actor_user_id=42,
        ))
        assert result.category == QuestionCategory.KNOWLEDGE
        assert result.should_use_knowledge is True
        assert result.should_use_system_context is False
        # Knowledge questions don't trigger context building
        assert result.context_bundle is None

    def test_system_question_builds_context(self):
        from app.services.orchestration.question_classifier import QuestionCategory

        orch = self._make_orchestrator()
        result = _run(orch.evaluate(
            question="Tổng số hồ sơ đang chờ duyệt là bao nhiêu?",
            tenant_id="t1",
            actor_user_id=42,
        ))
        assert result.category == QuestionCategory.SYSTEM
        assert result.should_use_system_context is True
        assert result.context_bundle is not None
        assert result.context_bundle.source == "mock"

    def test_access_question(self):
        from app.services.orchestration.question_classifier import QuestionCategory

        orch = self._make_orchestrator()
        result = _run(orch.evaluate(
            question="Tôi có quyền xem được gì?",
            tenant_id="t1",
            actor_user_id=42,
        ))
        assert result.category == QuestionCategory.ACCESS
        assert result.should_use_access_context is True
        assert result.context_bundle is not None

    def test_unknown_question(self):
        from app.services.orchestration.question_classifier import QuestionCategory

        orch = self._make_orchestrator()
        result = _run(orch.evaluate(
            question="Xin chào bạn",
            tenant_id="t1",
            actor_user_id=42,
        ))
        assert result.category == QuestionCategory.UNKNOWN
        assert result.should_use_knowledge is True

    def test_skip_context_build(self):
        orch = self._make_orchestrator()
        result = _run(orch.evaluate(
            question="Tổng số hồ sơ là bao nhiêu?",
            tenant_id="t1",
            actor_user_id=42,
            build_context=False,
        ))
        # Should classify but NOT build context
        assert result.context_bundle is None

    def test_no_builder_configured(self):
        from app.services.orchestration.system_context_orchestrator import (
            SystemContextOrchestrator,
        )

        orch = SystemContextOrchestrator()  # no context_builder
        result = _run(orch.evaluate(
            question="Tổng số hồ sơ là bao nhiêu?",
            tenant_id="t1",
            actor_user_id=42,
        ))
        assert result.context_bundle is None
        assert any("no_context_builder" in n for n in result.notes)

    def test_mixed_question(self):
        from app.services.orchestration.question_classifier import QuestionCategory

        orch = self._make_orchestrator()
        result = _run(orch.evaluate(
            question="Tổng số hồ sơ theo quy định mới nhất là bao nhiêu?",
            tenant_id="t1",
            actor_user_id=42,
        ))
        assert result.category == QuestionCategory.MIXED
        assert result.should_use_knowledge is True
        assert result.should_use_system_context is True
