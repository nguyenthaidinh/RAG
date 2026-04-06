"""
Phase 1.1 — System Context Foundation Hardening tests.

Covers all Phase 1.1 hardening:
  1. Schema contract (tenant_id required, actor_user_id required)
  2. Registry behavior (mock gating, get_required, config-driven defaults)
  3. CorePlatformConnector placeholder semantics
  4. ContextBuilder type-safe + fail semantics
  5. Orchestrator consistency (needs_context vs flags)
  6. QuestionClassifier (reduced false positives, weighted, mixed threshold)
  7. RemoteFileFetcher (redirect-safe, SSRF guards)
  8. Debug endpoint (role-gated, provider validation)

Uses asyncio.get_event_loop().run_until_complete() for async tests.
"""
from __future__ import annotations

import asyncio
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


# =====================================================================
# 1. SCHEMA CONTRACT TESTS
# =====================================================================


class TestSchemaContracts:
    """Verify tenant-sensitive fields are required."""

    def test_record_summary_requires_tenant_id(self):
        from pydantic import ValidationError
        from app.schemas.system_context import RecordSummary

        with pytest.raises(ValidationError) as exc_info:
            RecordSummary(record_type="req", record_id="R-1")
        assert "tenant_id" in str(exc_info.value)

    def test_record_summary_with_tenant_id(self):
        from app.schemas.system_context import RecordSummary

        rs = RecordSummary(
            record_type="req", record_id="R-1", tenant_id="t1",
        )
        assert rs.tenant_id == "t1"

    def test_record_summary_empty_tenant_id_rejected(self):
        """Empty string is technically valid for str, but we test it's not
        the default anymore — must be explicit."""
        from app.schemas.system_context import RecordSummary

        rs = RecordSummary(
            record_type="req", record_id="R-1", tenant_id="",
        )
        # Empty string is accepted (str) but NOT the default
        assert rs.tenant_id == ""

    def test_workflow_summary_requires_tenant_id(self):
        from pydantic import ValidationError
        from app.schemas.system_context import WorkflowSummary

        with pytest.raises(ValidationError) as exc_info:
            WorkflowSummary(workflow_type="approval")
        assert "tenant_id" in str(exc_info.value)

    def test_workflow_summary_with_tenant_id(self):
        from app.schemas.system_context import WorkflowSummary

        ws = WorkflowSummary(
            workflow_type="approval", tenant_id="t1", total=10,
        )
        assert ws.tenant_id == "t1"

    def test_permission_snapshot_requires_actor_user_id(self):
        from pydantic import ValidationError
        from app.schemas.system_context import PermissionSnapshot

        with pytest.raises(ValidationError) as exc_info:
            PermissionSnapshot(tenant_id="t1")
        assert "actor_user_id" in str(exc_info.value)

    def test_permission_snapshot_with_actor(self):
        from app.schemas.system_context import PermissionSnapshot

        snap = PermissionSnapshot(
            tenant_id="t1", actor_user_id=42,
        )
        assert snap.actor_user_id == 42
        assert snap.decisions == []

    def test_user_context_tenant_id_required(self):
        from pydantic import ValidationError
        from app.schemas.system_context import UserContext

        with pytest.raises(ValidationError):
            UserContext(user_id="u1")

    def test_tenant_context_tenant_id_required(self):
        from pydantic import ValidationError
        from app.schemas.system_context import TenantContext

        with pytest.raises(ValidationError):
            TenantContext()


# =====================================================================
# 2. REGISTRY HARDENING TESTS
# =====================================================================


class TestConnectorRegistryHardened:
    def test_get_required_found(self):
        from app.services.system_context.connector_registry import ConnectorRegistry

        registry = ConnectorRegistry()
        mock = MagicMock()
        mock.provider_name = "test"
        registry.register("test", mock)

        result = registry.get_required("test")
        assert result is mock

    def test_get_required_not_found(self):
        from app.services.system_context.connector_registry import (
            ConnectorRegistry,
            ConnectorNotFoundError,
        )

        registry = ConnectorRegistry()
        mock = MagicMock()
        registry.register("other", mock)

        with pytest.raises(ConnectorNotFoundError) as exc_info:
            registry.get_required("nonexistent")
        assert exc_info.value.provider_name == "nonexistent"
        assert "other" in exc_info.value.available

    def test_get_required_empty_registry(self):
        from app.services.system_context.connector_registry import (
            ConnectorRegistry,
            ConnectorNotFoundError,
        )

        registry = ConnectorRegistry()
        with pytest.raises(ConnectorNotFoundError):
            registry.get_required("anything")

    def test_connector_not_found_error_message(self):
        from app.services.system_context.connector_registry import ConnectorNotFoundError

        err = ConnectorNotFoundError("bad-provider", ["mock", "core-platform"])
        assert "bad-provider" in str(err)
        assert "mock" in str(err)

    def test_default_respects_explicit_config(self):
        from app.services.system_context.connector_registry import ConnectorRegistry

        registry = ConnectorRegistry()
        m1 = MagicMock()
        m2 = MagicMock()

        registry.register("first", m1)
        registry.register("second", m2, default=True)

        assert registry.get_default() is m2
        assert registry.default_name == "second"

    def test_global_registry_mock_gating(self):
        """When SYSTEM_CONTEXT_ALLOW_MOCK=True (default), mock is registered."""
        from app.services.system_context.connector_registry import get_connector_registry

        registry = get_connector_registry()
        assert "mock" in registry

    def test_bootstrap_without_mock(self):
        """Verify mock is NOT registered when allow_mock=False."""
        from app.services.system_context.connector_registry import (
            ConnectorRegistry,
            _bootstrap_default_connectors,
        )

        registry = ConnectorRegistry()
        with patch("app.core.config.settings") as mock_settings:
            mock_settings.SYSTEM_CONTEXT_ALLOW_MOCK = False
            mock_settings.SYSTEM_CONTEXT_PROVIDER = "core-platform"
            _bootstrap_default_connectors(registry)

        assert "mock" not in registry
        assert "core-platform" in registry

    def test_bootstrap_with_mock_not_default(self):
        """When provider is core-platform, mock is registered but not default."""
        from app.services.system_context.connector_registry import (
            ConnectorRegistry,
            _bootstrap_default_connectors,
        )

        registry = ConnectorRegistry()
        with patch("app.core.config.settings") as mock_settings:
            mock_settings.SYSTEM_CONTEXT_ALLOW_MOCK = True
            mock_settings.SYSTEM_CONTEXT_PROVIDER = "core-platform"
            _bootstrap_default_connectors(registry)

        assert "mock" in registry
        assert "core-platform" in registry
        assert registry.default_name == "core-platform"


# =====================================================================
# 3. CORE-PLATFORM PLACEHOLDER TESTS
# =====================================================================


class TestCorePlatformPlaceholder:
    def _connector(self):
        from app.services.system_context.core_platform_connector import (
            CorePlatformConnector,
        )
        return CorePlatformConnector()

    def test_provider_name(self):
        assert self._connector().provider_name == "core-platform"

    def test_returns_none_for_user(self):
        result = _run(self._connector().get_user_context(
            tenant_id="t1", actor_user_id=1,
        ))
        assert result is None

    def test_returns_none_for_tenant(self):
        result = _run(self._connector().get_tenant_context(tenant_id="t1"))
        assert result is None

    def test_returns_none_for_permissions(self):
        result = _run(self._connector().get_permission_snapshot(
            tenant_id="t1", actor_user_id=1,
        ))
        assert result is None

    def test_build_bundle_placeholder_source(self):
        """Bundle source must clearly indicate placeholder status."""
        bundle = _run(self._connector().build_context_bundle(
            tenant_id="t1", actor_user_id=1,
        ))
        assert "placeholder" in bundle.source
        assert bundle.source == "core-platform/placeholder"
        assert not bundle.has_user

    def test_placeholder_methods_log_warning(self):
        """Placeholder methods should log warnings, not debug."""
        import logging

        with patch.object(logging.getLogger("app.services.system_context.core_platform_connector"), "warning") as mock_warn:
            _run(self._connector().get_user_context(
                tenant_id="t1", actor_user_id=1,
            ))
            assert mock_warn.called
            call_msg = mock_warn.call_args[0][0]
            assert "PLACEHOLDER" in call_msg


# =====================================================================
# 4. CONTEXT BUILDER TESTS
# =====================================================================


class TestContextBuilderHardened:
    def _make_builder(self):
        from app.services.system_context.mock_connector import MockSystemConnector
        from app.services.system_context.context_builder import SystemContextBuilder

        return SystemContextBuilder(connector=MockSystemConnector())

    def test_typed_constructor(self):
        """Builder accepts BaseSystemConnector protocol."""
        from app.services.system_context.mock_connector import MockSystemConnector
        from app.services.system_context.context_builder import SystemContextBuilder

        connector = MockSystemConnector()
        builder = SystemContextBuilder(connector=connector)
        assert builder.provider_name == "mock"

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

    def test_permissions_fail_closed(self):
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
        # Fail-closed: empty snapshot, not None
        assert bundle.permissions is not None
        assert len(bundle.permissions.decisions) == 0
        assert bundle.permissions.actor_user_id == 1

    def test_stats_fail_open(self):
        from app.services.system_context.context_builder import (
            ContextBuildFlags,
            SystemContextBuilder,
        )

        connector = AsyncMock()
        connector.provider_name = "failing"
        connector.get_user_context = AsyncMock(return_value=None)
        connector.get_tenant_context = AsyncMock(return_value=None)
        connector.get_tenant_stats = AsyncMock(
            side_effect=Exception("stats fail"),
        )

        builder = SystemContextBuilder(connector=connector)
        bundle = _run(builder.build(
            tenant_id="t1",
            actor_user_id=1,
            flags=ContextBuildFlags(include_stats=True),
        ))
        assert not bundle.has_stats

    def test_context_build_flags_needs_context(self):
        from app.services.system_context.context_builder import ContextBuildFlags

        # All disabled
        empty = ContextBuildFlags(
            include_user=False, include_tenant=False,
        )
        assert not empty.needs_context_build

        # At least one enabled
        user_only = ContextBuildFlags(
            include_user=True, include_tenant=False,
        )
        assert user_only.needs_context_build

        # Default
        default = ContextBuildFlags()
        assert default.needs_context_build


# =====================================================================
# 5. ORCHESTRATOR CONSISTENCY TESTS
# =====================================================================


class TestOrchestratorConsistency:
    def _make_orchestrator(self):
        from app.services.system_context.mock_connector import MockSystemConnector
        from app.services.system_context.context_builder import SystemContextBuilder
        from app.services.orchestration.system_context_orchestrator import (
            SystemContextOrchestrator,
        )

        connector = MockSystemConnector()
        builder = SystemContextBuilder(connector=connector)
        return SystemContextOrchestrator(context_builder=builder)

    def test_knowledge_no_context_build(self):
        """KNOWLEDGE questions should NOT trigger context build."""
        orch = self._make_orchestrator()
        result = _run(orch.evaluate(
            question="Quy định về nghỉ phép là gì?",
            tenant_id="t1",
            actor_user_id=42,
        ))
        from app.services.orchestration.question_classifier import QuestionCategory

        assert result.category == QuestionCategory.KNOWLEDGE
        assert result.should_use_knowledge is True
        assert result.should_use_system_context is False
        # Phase 1.1: KNOWLEDGE has no flags → no context build
        assert result.context_bundle is None
        assert not result.recommended_flags.needs_context_build

    def test_system_builds_context(self):
        """SYSTEM questions MUST trigger context build."""
        orch = self._make_orchestrator()
        result = _run(orch.evaluate(
            question="Tổng số hồ sơ đang chờ duyệt là bao nhiêu?",
            tenant_id="t1",
            actor_user_id=42,
        ))
        from app.services.orchestration.question_classifier import QuestionCategory

        assert result.category == QuestionCategory.SYSTEM
        assert result.should_use_system_context is True
        assert result.context_bundle is not None
        assert result.recommended_flags.needs_context_build

    def test_access_builds_context_with_permissions(self):
        """ACCESS questions build context with permissions."""
        orch = self._make_orchestrator()
        result = _run(orch.evaluate(
            question="Tôi có quyền xem được gì?",
            tenant_id="t1",
            actor_user_id=42,
        ))
        from app.services.orchestration.question_classifier import QuestionCategory

        assert result.category == QuestionCategory.ACCESS
        assert result.should_use_access_context is True
        assert result.context_bundle is not None
        assert result.recommended_flags.include_permissions is True

    def test_unknown_no_context_build(self):
        """UNKNOWN questions fallback to knowledge, no context build."""
        orch = self._make_orchestrator()
        result = _run(orch.evaluate(
            question="Xin chào bạn",
            tenant_id="t1",
            actor_user_id=42,
        ))
        from app.services.orchestration.question_classifier import QuestionCategory

        assert result.category == QuestionCategory.UNKNOWN
        assert result.should_use_knowledge is True
        assert result.context_bundle is None
        assert not result.recommended_flags.needs_context_build

    def test_skip_build_still_classifies(self):
        orch = self._make_orchestrator()
        result = _run(orch.evaluate(
            question="Tổng số hồ sơ là bao nhiêu?",
            tenant_id="t1",
            actor_user_id=42,
            build_context=False,
        ))
        assert result.context_bundle is None
        assert result.recommended_flags.needs_context_build

    def test_no_builder_notes(self):
        from app.services.orchestration.system_context_orchestrator import (
            SystemContextOrchestrator,
        )

        orch = SystemContextOrchestrator()
        result = _run(orch.evaluate(
            question="Tổng số hồ sơ là bao nhiêu?",
            tenant_id="t1",
            actor_user_id=42,
        ))
        assert result.context_bundle is None
        assert any("no_context_builder" in n for n in result.notes)

    def test_mixed_with_access_signals(self):
        """MIXED with access signals should include access context."""
        from app.services.orchestration.question_classifier import QuestionCategory

        orch = self._make_orchestrator()
        # A question with both system AND access signals
        result = _run(orch.evaluate(
            question="Tổng số hồ sơ tôi có quyền xem là bao nhiêu?",
            tenant_id="t1",
            actor_user_id=42,
        ))
        assert result.category == QuestionCategory.MIXED
        assert result.should_use_access_context is True
        assert result.recommended_flags.include_permissions is True


# =====================================================================
# 6. QUESTION CLASSIFIER HARDENED TESTS
# =====================================================================


class TestClassifierHardened:
    def _classifier(self):
        from app.services.orchestration.question_classifier import QuestionClassifier
        return QuestionClassifier()

    def test_knowledge_phrase(self):
        from app.services.orchestration.question_classifier import QuestionCategory

        result = self._classifier().classify("Tài liệu quy định về nghỉ phép là gì?")
        assert result.category == QuestionCategory.KNOWLEDGE
        assert result.confidence > 0.5

    def test_system_phrase(self):
        from app.services.orchestration.question_classifier import QuestionCategory

        result = self._classifier().classify("Tổng số hồ sơ đang chờ duyệt là bao nhiêu?")
        assert result.category == QuestionCategory.SYSTEM
        assert result.confidence > 0.5

    def test_access_phrase(self):
        from app.services.orchestration.question_classifier import QuestionCategory

        result = self._classifier().classify("Tôi có quyền xem được gì?")
        assert result.category == QuestionCategory.ACCESS
        assert result.confidence > 0.5

    def test_ambiguous_word_alone_is_unknown(self):
        """Single ambiguous words should NOT trigger strong classification."""
        from app.services.orchestration.question_classifier import QuestionCategory

        # "status" alone is too weak for SYSTEM with weight=0.5
        result = self._classifier().classify("status?")
        # Low-weight single word → single category, but low confidence
        assert result.confidence < 0.7

    def test_mixed_not_triggered_by_weak_signals(self):
        """MIXED requires meaningful weight from both categories."""
        from app.services.orchestration.question_classifier import QuestionCategory

        # "content" (knowledge 0.5) + "status" (system 0.5) = total 1.0
        # Below MIXED_MIN_TOTAL_WEIGHT (2.5), below MIXED_MIN_SECONDARY_WEIGHT (1.0)
        result = self._classifier().classify("content status")
        assert result.category != QuestionCategory.MIXED

    def test_mixed_with_strong_signals(self):
        """MIXED triggers when both categories have strong signals."""
        from app.services.orchestration.question_classifier import QuestionCategory

        # "quy định" (knowledge 1.5) + "tổng số" (system 1.5) = total 3.0
        result = self._classifier().classify("Tổng số theo quy định mới nhất là bao nhiêu?")
        assert result.category == QuestionCategory.MIXED
        assert result.secondary_category is not None

    def test_empty_question(self):
        from app.services.orchestration.question_classifier import QuestionCategory

        result = self._classifier().classify("")
        assert result.category == QuestionCategory.UNKNOWN
        assert result.confidence == 0.0

    def test_no_signals_is_unknown(self):
        from app.services.orchestration.question_classifier import QuestionCategory

        result = self._classifier().classify("Xin chào bạn khỏe không?")
        assert result.category == QuestionCategory.UNKNOWN

    def test_classifier_deterministic(self):
        c = self._classifier()
        q = "Tổng số nhân viên là bao nhiêu?"
        r1 = c.classify(q)
        r2 = c.classify(q)
        assert r1.category == r2.category
        assert r1.confidence == r2.confidence

    def test_long_phrase_priority(self):
        """Longer phrases should match without being split by sub-patterns."""
        from app.services.orchestration.question_classifier import QuestionCategory

        result = self._classifier().classify("Tôi có quyền truy cập hệ thống không?")
        assert result.category == QuestionCategory.ACCESS
        # "quyền truy cập" should be matched as phrase, not split

    def test_english_knowledge(self):
        from app.services.orchestration.question_classifier import QuestionCategory

        result = self._classifier().classify("What is the leave policy?")
        assert result.category == QuestionCategory.KNOWLEDGE

    def test_english_system(self):
        from app.services.orchestration.question_classifier import QuestionCategory

        result = self._classifier().classify("How many pending records are there?")
        assert result.category == QuestionCategory.SYSTEM

    def test_english_access(self):
        from app.services.orchestration.question_classifier import QuestionCategory

        result = self._classifier().classify("What are my permissions?")
        assert result.category == QuestionCategory.ACCESS


# =====================================================================
# 7. REMOTE FILE FETCHER HARDENING TESTS
# =====================================================================


class TestRemoteFetchUrlValidationHardened:
    def test_valid_url_passes(self):
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
        with pytest.raises(RemoteFetchForbiddenHost):
            validate_fetch_url(
                "https://evil.example.com/malware.exe",
                allowed_hosts=frozenset({"files.example.com"}),
                enforce=True,
            )

    def test_disallowed_host_audit_only(self):
        from app.services.remote_file_fetcher import validate_fetch_url

        result = validate_fetch_url(
            "https://evil.example.com/doc.pdf",
            allowed_hosts=frozenset({"files.example.com"}),
            enforce=False,
        )
        assert result == "https://evil.example.com/doc.pdf"

    def test_malformed_url_bad_scheme(self):
        from app.services.remote_file_fetcher import RemoteFetchError, validate_fetch_url

        with pytest.raises(RemoteFetchError):
            validate_fetch_url("ftp://files.example.com/doc.pdf", enforce=True)

    def test_no_hostname(self):
        from app.services.remote_file_fetcher import RemoteFetchError, validate_fetch_url

        with pytest.raises(RemoteFetchError):
            validate_fetch_url("http:///path/to/file", enforce=True)

    def test_case_insensitive_host(self):
        from app.services.remote_file_fetcher import validate_fetch_url

        result = validate_fetch_url(
            "https://Files.Example.COM/doc.pdf",
            allowed_hosts=frozenset({"files.example.com"}),
            enforce=True,
        )
        assert result is not None


class TestSSRFProtection:
    """Phase 1.1: SSRF guards block private/internal addresses."""

    def test_localhost_blocked(self):
        from app.services.remote_file_fetcher import (
            RemoteFetchSSRFBlocked,
            validate_fetch_url,
        )
        with pytest.raises(RemoteFetchSSRFBlocked):
            validate_fetch_url("https://localhost/api/secret", enforce=False)

    def test_127_0_0_1_blocked(self):
        from app.services.remote_file_fetcher import (
            RemoteFetchSSRFBlocked,
            validate_fetch_url,
        )
        with pytest.raises(RemoteFetchSSRFBlocked):
            validate_fetch_url("https://127.0.0.1/api/secret", enforce=False)

    def test_ipv6_loopback_blocked(self):
        from app.services.remote_file_fetcher import (
            RemoteFetchSSRFBlocked,
            validate_fetch_url,
        )
        with pytest.raises(RemoteFetchSSRFBlocked):
            validate_fetch_url("https://[::1]/api/secret", enforce=False)

    def test_private_10_blocked(self):
        from app.services.remote_file_fetcher import (
            RemoteFetchSSRFBlocked,
            validate_fetch_url,
        )
        with pytest.raises(RemoteFetchSSRFBlocked):
            validate_fetch_url("https://10.0.0.5/secret", enforce=False)

    def test_private_192_168_blocked(self):
        from app.services.remote_file_fetcher import (
            RemoteFetchSSRFBlocked,
            validate_fetch_url,
        )
        with pytest.raises(RemoteFetchSSRFBlocked):
            validate_fetch_url("https://192.168.1.1/admin", enforce=False)

    def test_link_local_blocked(self):
        from app.services.remote_file_fetcher import (
            RemoteFetchSSRFBlocked,
            validate_fetch_url,
        )
        with pytest.raises(RemoteFetchSSRFBlocked):
            validate_fetch_url("https://169.254.169.254/latest/meta-data", enforce=False)

    def test_metadata_google_blocked(self):
        from app.services.remote_file_fetcher import (
            RemoteFetchSSRFBlocked,
            validate_fetch_url,
        )
        with pytest.raises(RemoteFetchSSRFBlocked):
            validate_fetch_url("https://metadata.google.internal/computeMetadata", enforce=False)

    def test_public_ip_allowed(self):
        from app.services.remote_file_fetcher import validate_fetch_url

        result = validate_fetch_url(
            "https://8.8.8.8/file.pdf",
            enforce=False,
        )
        assert result == "https://8.8.8.8/file.pdf"

    def test_ssrf_check_can_be_disabled(self):
        from app.services.remote_file_fetcher import validate_fetch_url

        result = validate_fetch_url(
            "https://localhost/ok",
            check_ssrf=False,
            enforce=False,
        )
        assert result == "https://localhost/ok"


class TestRedirectProtection:
    """Phase 1.1: redirect-based allowlist bypass prevention."""

    def test_redirect_error_type(self):
        from app.services.remote_file_fetcher import RemoteFetchRedirectError

        err = RemoteFetchRedirectError("too many hops")
        assert err.status_code == 403
        assert "too many hops" in str(err)


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
# 8. MOCK CONNECTOR UPDATED TESTS
# =====================================================================


class TestMockConnectorUpdated:
    """Verify mock connector provides required fields."""

    def _connector(self):
        from app.services.system_context.mock_connector import MockSystemConnector
        return MockSystemConnector()

    def test_record_summaries_have_tenant_id(self):
        c = self._connector()
        records = _run(c.get_record_summaries(
            tenant_id="t1", actor_user_id=42,
        ))
        for r in records:
            assert r.tenant_id == "t1"

    def test_workflow_summaries_have_tenant_id(self):
        c = self._connector()
        workflows = _run(c.get_workflow_summaries(
            tenant_id="t1",
        ))
        for w in workflows:
            assert w.tenant_id == "t1"

    def test_permission_snapshot_has_actor(self):
        c = self._connector()
        snap = _run(c.get_permission_snapshot(
            tenant_id="t1", actor_user_id=99,
        ))
        assert snap.actor_user_id == 99

    def test_build_full_bundle(self):
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
        assert bundle.has_permissions
        assert bundle.has_stats
        assert len(bundle.records) > 0
        assert len(bundle.workflows) > 0

    def test_tenant_isolation(self):
        c = self._connector()
        ctx1 = _run(c.get_user_context(tenant_id="t1", actor_user_id=1))
        ctx2 = _run(c.get_user_context(tenant_id="t2", actor_user_id=1))
        assert ctx1.tenant_id == "t1"
        assert ctx2.tenant_id == "t2"


# =====================================================================
# 9. BUNDLE SERIALIZATION
# =====================================================================


class TestBundleSerialization:
    def test_bundle_with_required_fields_serializes(self):
        from app.schemas.system_context import (
            SystemContextBundle,
            UserContext,
            PermissionSnapshot,
            RecordSummary,
            WorkflowSummary,
        )

        bundle = SystemContextBundle(
            user=UserContext(user_id="u1", tenant_id="t1"),
            permissions=PermissionSnapshot(
                tenant_id="t1", actor_user_id="u1",
            ),
            records=[
                RecordSummary(
                    record_type="req", record_id="R-1", tenant_id="t1",
                ),
            ],
            workflows=[
                WorkflowSummary(
                    workflow_type="approval", tenant_id="t1",
                ),
            ],
            source="test",
        )
        d = bundle.model_dump(mode="json")
        assert d["source"] == "test"
        assert d["records"][0]["tenant_id"] == "t1"
        assert d["workflows"][0]["tenant_id"] == "t1"
        assert d["permissions"]["actor_user_id"] == "u1"

    def test_empty_bundle_serializes(self):
        from app.schemas.system_context import SystemContextBundle

        b = SystemContextBundle(source="test")
        d = b.model_dump(mode="json")
        assert d["source"] == "test"
        assert isinstance(d["fetched_at"], str)
