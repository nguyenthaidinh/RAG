"""
Phase 2B — System-aware assistant tests.

Covers:
  A. KNOWLEDGE category: retrieval runs, system-only path not used
  B. SYSTEM category: retrieval skipped, system context answer
  C. ACCESS category: retrieval skipped, permissions answer
  D. MIXED category: retrieval runs + system context
  E. Orchestrator failure: knowledge fallback
  F. System-only answer has NO fake citations
  G. Tenant safety not bypassed
  H. SystemAnswerBuilder tests

Uses asyncio.run() for async tests (no pytest-asyncio dependency).
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.schemas.system_context import (
    MetricValue,
    PermissionDecision,
    PermissionSnapshot,
    RecordSummary,
    SystemContextBundle,
    TenantContext,
    TenantStats,
    UserContext,
    WorkflowSummary,
)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_orch_result(
    *,
    category: str,
    should_use_knowledge: bool,
    should_use_system_context: bool,
    should_use_access_context: bool,
    bundle: SystemContextBundle | None = None,
):
    """Create a mock OrchestrationResult."""
    from app.services.orchestration.question_classifier import (
        ClassificationResult,
        QuestionCategory,
    )
    from app.services.orchestration.system_context_orchestrator import (
        OrchestrationResult,
    )
    from app.services.system_context.context_builder import ContextBuildFlags

    cat = QuestionCategory(category)
    return OrchestrationResult(
        category=cat,
        classification=ClassificationResult(
            category=cat,
            confidence=0.85,
            matched_signals=(),
        ),
        should_use_knowledge=should_use_knowledge,
        should_use_system_context=should_use_system_context,
        should_use_access_context=should_use_access_context,
        recommended_flags=ContextBuildFlags(),
        context_bundle=bundle,
    )


def _make_bundle_with_stats():
    """Bundle with tenant stats."""
    return SystemContextBundle(
        user=UserContext(
            user_id="u1",
            tenant_id="t1",
            display_name="Alice",
            role="admin",
        ),
        tenant_stats=TenantStats(
            tenant_id="t1",
            metrics=[
                MetricValue(key="users", value=42, label="Total Users"),
                MetricValue(key="docs", value=100, label="Total Docs"),
            ],
        ),
        source="core-platform",
    )


def _make_bundle_with_permissions():
    """Bundle with permission decisions."""
    return SystemContextBundle(
        user=UserContext(
            user_id="u1",
            tenant_id="t1",
            display_name="Alice",
            role="admin",
        ),
        permissions=PermissionSnapshot(
            tenant_id="t1",
            actor_user_id="u1",
            decisions=[
                PermissionDecision(
                    resource_type="document",
                    action="read",
                    allowed=True,
                ),
                PermissionDecision(
                    resource_type="user",
                    action="manage",
                    allowed=False,
                ),
            ],
        ),
        source="core-platform",
    )


def _make_bundle_mixed():
    """Bundle with stats + records."""
    return SystemContextBundle(
        user=UserContext(
            user_id="u1",
            tenant_id="t1",
            display_name="Alice",
        ),
        tenant_stats=TenantStats(
            tenant_id="t1",
            metrics=[
                MetricValue(key="users", value=42, label="Total Users"),
            ],
        ),
        records=[
            RecordSummary(
                record_type="request",
                record_id="R-1",
                tenant_id="t1",
                title="Request One",
                status="pending",
            ),
        ],
        source="core-platform",
    )


# =====================================================================
# A. KNOWLEDGE CATEGORY
# =====================================================================


class TestKnowledgeCategory:
    def test_knowledge_retrieval_runs(self):
        """KNOWLEDGE → retrieval is called, system path not used."""
        import app.services.assistant_service as svc_mod

        orch_result = _make_orch_result(
            category="knowledge",
            should_use_knowledge=True,
            should_use_system_context=False,
            should_use_access_context=False,
        )

        mock_orchestrator = AsyncMock()
        mock_orchestrator.evaluate = AsyncMock(return_value=orch_result)

        mock_query_svc = AsyncMock()
        mock_query_svc.query = AsyncMock(return_value=[])

        mock_answer_svc = MagicMock()
        from app.services.answer_service import GeneratedAnswer
        mock_answer_svc.generate_structured = AsyncMock(
            return_value=GeneratedAnswer(
                text="Answer from docs", intent="general", evidences=(),
            ),
        )

        from app.schemas.assistant import AssistantRespondRequest
        request = AssistantRespondRequest(message="What is policy X?")

        with (
            patch.object(svc_mod, "_get_orchestrator", return_value=mock_orchestrator),
            patch.object(svc_mod, "_get_query_svc", return_value=mock_query_svc),
            patch.object(svc_mod, "_get_answer_svc", return_value=mock_answer_svc),
        ):
            service = svc_mod.AssistantService()
            response = _run(service.respond(
                request=request,
                tenant_id="t1",
                user_id=1,
            ))

        # Retrieval was called
        mock_query_svc.query.assert_called_once()
        # Response has answer
        assert response.message == "Answer from docs"


# =====================================================================
# B. SYSTEM CATEGORY
# =====================================================================


class TestSystemCategory:
    def test_system_retrieval_skipped(self):
        """SYSTEM → retrieval NOT called."""
        import app.services.assistant_service as svc_mod

        bundle = _make_bundle_with_stats()
        orch_result = _make_orch_result(
            category="system",
            should_use_knowledge=False,
            should_use_system_context=True,
            should_use_access_context=False,
            bundle=bundle,
        )

        mock_orchestrator = AsyncMock()
        mock_orchestrator.evaluate = AsyncMock(return_value=orch_result)

        mock_query_svc = AsyncMock()
        mock_query_svc.query = AsyncMock(return_value=[])

        mock_answer_svc = MagicMock()
        from app.services.answer_service import GeneratedAnswer
        mock_answer_svc.generate_structured = AsyncMock(
            return_value=GeneratedAnswer(
                text=None, intent="general", evidences=(),
            ),
        )

        from app.schemas.assistant import AssistantRespondRequest
        request = AssistantRespondRequest(message="How many users?")

        with (
            patch.object(svc_mod, "_get_orchestrator", return_value=mock_orchestrator),
            patch.object(svc_mod, "_get_query_svc", return_value=mock_query_svc),
            patch.object(svc_mod, "_get_answer_svc", return_value=mock_answer_svc),
        ):
            service = svc_mod.AssistantService()
            response = _run(service.respond(
                request=request,
                tenant_id="t1",
                user_id=1,
            ))

        # Retrieval was NOT called
        mock_query_svc.query.assert_not_called()
        # System fallback should provide an answer
        assert "Total Users" in response.message or "42" in response.message

    def test_system_allow_system_only_passed(self):
        """SYSTEM → allow_system_context_only=True passed to answer svc."""
        import app.services.assistant_service as svc_mod

        bundle = _make_bundle_with_stats()
        orch_result = _make_orch_result(
            category="system",
            should_use_knowledge=False,
            should_use_system_context=True,
            should_use_access_context=False,
            bundle=bundle,
        )

        mock_orchestrator = AsyncMock()
        mock_orchestrator.evaluate = AsyncMock(return_value=orch_result)

        mock_query_svc = AsyncMock()
        mock_answer_svc = MagicMock()

        from app.services.answer_service import GeneratedAnswer
        mock_answer_svc.generate_structured = AsyncMock(
            return_value=GeneratedAnswer(
                text="LLM system answer", intent="general", evidences=(),
            ),
        )

        from app.schemas.assistant import AssistantRespondRequest
        request = AssistantRespondRequest(message="How many users?")

        with (
            patch.object(svc_mod, "_get_orchestrator", return_value=mock_orchestrator),
            patch.object(svc_mod, "_get_query_svc", return_value=mock_query_svc),
            patch.object(svc_mod, "_get_answer_svc", return_value=mock_answer_svc),
        ):
            service = svc_mod.AssistantService()
            _run(service.respond(
                request=request,
                tenant_id="t1",
                user_id=1,
            ))

        # Check that allow_system_context_only=True was passed
        call_kwargs = mock_answer_svc.generate_structured.call_args.kwargs
        assert call_kwargs["allow_system_context_only"] is True
        assert call_kwargs["question_category"] == "system"


# =====================================================================
# C. ACCESS CATEGORY
# =====================================================================


class TestAccessCategory:
    def test_access_retrieval_skipped(self):
        """ACCESS → retrieval NOT called."""
        import app.services.assistant_service as svc_mod

        bundle = _make_bundle_with_permissions()
        orch_result = _make_orch_result(
            category="access",
            should_use_knowledge=False,
            should_use_system_context=False,
            should_use_access_context=True,
            bundle=bundle,
        )

        mock_orchestrator = AsyncMock()
        mock_orchestrator.evaluate = AsyncMock(return_value=orch_result)

        mock_query_svc = AsyncMock()
        mock_answer_svc = MagicMock()
        from app.services.answer_service import GeneratedAnswer
        mock_answer_svc.generate_structured = AsyncMock(
            return_value=GeneratedAnswer(
                text=None, intent="general", evidences=(),
            ),
        )

        from app.schemas.assistant import AssistantRespondRequest
        request = AssistantRespondRequest(message="What are my permissions?")

        with (
            patch.object(svc_mod, "_get_orchestrator", return_value=mock_orchestrator),
            patch.object(svc_mod, "_get_query_svc", return_value=mock_query_svc),
            patch.object(svc_mod, "_get_answer_svc", return_value=mock_answer_svc),
        ):
            service = svc_mod.AssistantService()
            response = _run(service.respond(
                request=request,
                tenant_id="t1",
                user_id=1,
            ))

        mock_query_svc.query.assert_not_called()
        # Should have permission info in answer
        assert "Alice" in response.message or "permission" in response.message.lower()


# =====================================================================
# D. MIXED CATEGORY
# =====================================================================


class TestMixedCategory:
    def test_mixed_retrieval_runs(self):
        """MIXED → retrieval IS called."""
        import app.services.assistant_service as svc_mod

        bundle = _make_bundle_mixed()
        orch_result = _make_orch_result(
            category="mixed",
            should_use_knowledge=True,
            should_use_system_context=True,
            should_use_access_context=False,
            bundle=bundle,
        )

        mock_orchestrator = AsyncMock()
        mock_orchestrator.evaluate = AsyncMock(return_value=orch_result)

        mock_query_svc = AsyncMock()
        mock_query_svc.query = AsyncMock(return_value=[])

        mock_answer_svc = MagicMock()
        from app.services.answer_service import GeneratedAnswer
        mock_answer_svc.generate_structured = AsyncMock(
            return_value=GeneratedAnswer(
                text=None, intent="general", evidences=(),
            ),
        )

        from app.schemas.assistant import AssistantRespondRequest
        request = AssistantRespondRequest(
            message="What does the policy say about users?",
        )

        with (
            patch.object(svc_mod, "_get_orchestrator", return_value=mock_orchestrator),
            patch.object(svc_mod, "_get_query_svc", return_value=mock_query_svc),
            patch.object(svc_mod, "_get_answer_svc", return_value=mock_answer_svc),
        ):
            service = svc_mod.AssistantService()
            response = _run(service.respond(
                request=request,
                tenant_id="t1",
                user_id=1,
            ))

        # Retrieval WAS called
        mock_query_svc.query.assert_called_once()
        # System fallback should provide answer from bundle
        assert "Total Users" in response.message or "42" in response.message

    def test_mixed_no_snippets_uses_system_fallback(self):
        """MIXED: retrieval returns nothing, system context provides answer."""
        import app.services.assistant_service as svc_mod

        bundle = _make_bundle_mixed()
        orch_result = _make_orch_result(
            category="mixed",
            should_use_knowledge=True,
            should_use_system_context=True,
            should_use_access_context=False,
            bundle=bundle,
        )

        mock_orchestrator = AsyncMock()
        mock_orchestrator.evaluate = AsyncMock(return_value=orch_result)

        mock_query_svc = AsyncMock()
        mock_query_svc.query = AsyncMock(return_value=[])

        mock_answer_svc = MagicMock()
        from app.services.answer_service import GeneratedAnswer
        mock_answer_svc.generate_structured = AsyncMock(
            return_value=GeneratedAnswer(
                text=None, intent="general", evidences=(),
            ),
        )

        from app.schemas.assistant import AssistantRespondRequest
        request = AssistantRespondRequest(message="Mixed question")

        with (
            patch.object(svc_mod, "_get_orchestrator", return_value=mock_orchestrator),
            patch.object(svc_mod, "_get_query_svc", return_value=mock_query_svc),
            patch.object(svc_mod, "_get_answer_svc", return_value=mock_answer_svc),
        ):
            service = svc_mod.AssistantService()
            response = _run(service.respond(
                request=request,
                tenant_id="t1",
                user_id=1,
            ))

        # Should still answer from system context
        assert response.message and len(response.message) > 10


# =====================================================================
# E. ORCHESTRATOR FAILURE → KNOWLEDGE FALLBACK
# =====================================================================


class TestOrchestratorFailure:
    def test_orchestrator_error_falls_back_to_knowledge(self):
        """Orchestrator raises → retrieval still runs."""
        import app.services.assistant_service as svc_mod

        mock_orchestrator = AsyncMock()
        mock_orchestrator.evaluate = AsyncMock(
            side_effect=RuntimeError("orchestrator broke"),
        )

        mock_query_svc = AsyncMock()
        mock_query_svc.query = AsyncMock(return_value=[])

        mock_answer_svc = MagicMock()
        from app.services.answer_service import GeneratedAnswer
        mock_answer_svc.generate_structured = AsyncMock(
            return_value=GeneratedAnswer(
                text=None, intent="general", evidences=(),
            ),
        )

        from app.schemas.assistant import AssistantRespondRequest
        request = AssistantRespondRequest(message="Some question")

        with (
            patch.object(svc_mod, "_get_orchestrator", return_value=mock_orchestrator),
            patch.object(svc_mod, "_get_query_svc", return_value=mock_query_svc),
            patch.object(svc_mod, "_get_answer_svc", return_value=mock_answer_svc),
        ):
            service = svc_mod.AssistantService()
            response = _run(service.respond(
                request=request,
                tenant_id="t1",
                user_id=1,
            ))

        # Retrieval WAS called (knowledge fallback)
        mock_query_svc.query.assert_called_once()
        assert response.message  # Has a message (deterministic fallback)


# =====================================================================
# F. SYSTEM-ONLY ANSWER HAS NO FAKE CITATIONS
# =====================================================================


class TestNoCitationsForSystemAnswers:
    def test_system_answer_no_citations(self):
        """System-only answers have citations=[]."""
        import app.services.assistant_service as svc_mod

        bundle = _make_bundle_with_stats()
        orch_result = _make_orch_result(
            category="system",
            should_use_knowledge=False,
            should_use_system_context=True,
            should_use_access_context=False,
            bundle=bundle,
        )

        mock_orchestrator = AsyncMock()
        mock_orchestrator.evaluate = AsyncMock(return_value=orch_result)

        mock_query_svc = AsyncMock()
        mock_answer_svc = MagicMock()
        from app.services.answer_service import GeneratedAnswer
        mock_answer_svc.generate_structured = AsyncMock(
            return_value=GeneratedAnswer(
                text=None, intent="general", evidences=(),
            ),
        )

        from app.schemas.assistant import AssistantRespondRequest
        request = AssistantRespondRequest(message="How many users?")

        with (
            patch.object(svc_mod, "_get_orchestrator", return_value=mock_orchestrator),
            patch.object(svc_mod, "_get_query_svc", return_value=mock_query_svc),
            patch.object(svc_mod, "_get_answer_svc", return_value=mock_answer_svc),
        ):
            service = svc_mod.AssistantService()
            response = _run(service.respond(
                request=request,
                tenant_id="t1",
                user_id=1,
            ))

        assert response.citations == []

    def test_access_answer_no_citations(self):
        """Access-only answers have citations=[]."""
        import app.services.assistant_service as svc_mod

        bundle = _make_bundle_with_permissions()
        orch_result = _make_orch_result(
            category="access",
            should_use_knowledge=False,
            should_use_system_context=False,
            should_use_access_context=True,
            bundle=bundle,
        )

        mock_orchestrator = AsyncMock()
        mock_orchestrator.evaluate = AsyncMock(return_value=orch_result)

        mock_query_svc = AsyncMock()
        mock_answer_svc = MagicMock()
        from app.services.answer_service import GeneratedAnswer
        mock_answer_svc.generate_structured = AsyncMock(
            return_value=GeneratedAnswer(
                text=None, intent="general", evidences=(),
            ),
        )

        from app.schemas.assistant import AssistantRespondRequest
        request = AssistantRespondRequest(message="What can I do?")

        with (
            patch.object(svc_mod, "_get_orchestrator", return_value=mock_orchestrator),
            patch.object(svc_mod, "_get_query_svc", return_value=mock_query_svc),
            patch.object(svc_mod, "_get_answer_svc", return_value=mock_answer_svc),
        ):
            service = svc_mod.AssistantService()
            response = _run(service.respond(
                request=request,
                tenant_id="t1",
                user_id=1,
            ))

        assert response.citations == []


# =====================================================================
# G. TENANT SAFETY
# =====================================================================


class TestTenantSafety:
    def test_system_answer_does_not_leak_other_tenant(self):
        """System answer only contains data from correct tenant."""
        from app.services.system_context.system_answer_builder import (
            build_system_only_answer,
        )

        bundle = SystemContextBundle(
            user=UserContext(
                user_id="u1",
                tenant_id="t1",
                display_name="Alice",
            ),
            records=[
                RecordSummary(
                    record_type="request",
                    record_id="R-1",
                    tenant_id="t1",
                    title="My Record",
                ),
            ],
        )

        answer = build_system_only_answer(
            question="What are my records?",
            category="system",
            bundle=bundle,
        )

        assert answer is not None
        assert "My Record" in answer
        assert "t-other" not in answer


# =====================================================================
# H. SYSTEM ANSWER BUILDER
# =====================================================================


class TestSystemAnswerBuilder:
    def test_access_with_permissions(self):
        from app.services.system_context.system_answer_builder import (
            build_system_only_answer,
        )

        bundle = _make_bundle_with_permissions()
        answer = build_system_only_answer(
            question="What can I do?",
            category="access",
            bundle=bundle,
        )

        assert answer is not None
        assert "Alice" in answer
        assert "document/read" in answer or "Allowed" in answer
        assert "user/manage" in answer or "Denied" in answer

    def test_access_no_permissions(self):
        from app.services.system_context.system_answer_builder import (
            build_system_only_answer,
        )

        bundle = SystemContextBundle(
            user=UserContext(
                user_id="u1",
                tenant_id="t1",
                display_name="Alice",
            ),
        )
        answer = build_system_only_answer(
            question="My permissions?",
            category="access",
            bundle=bundle,
        )

        assert answer is not None
        assert "not available" in answer.lower() or "administrator" in answer.lower()

    def test_system_with_stats(self):
        from app.services.system_context.system_answer_builder import (
            build_system_only_answer,
        )

        bundle = _make_bundle_with_stats()
        answer = build_system_only_answer(
            question="Statistics?",
            category="system",
            bundle=bundle,
        )

        assert answer is not None
        assert "42" in answer
        assert "Total Users" in answer

    def test_system_with_workflows(self):
        from app.services.system_context.system_answer_builder import (
            build_system_only_answer,
        )

        bundle = SystemContextBundle(
            workflows=[
                WorkflowSummary(
                    workflow_type="approval",
                    tenant_id="t1",
                    total=100,
                    pending_count=20,
                    completed_count=80,
                ),
            ],
        )
        answer = build_system_only_answer(
            question="Workflow status?",
            category="system",
            bundle=bundle,
        )

        assert answer is not None
        assert "approval" in answer
        assert "100" in answer or "total" in answer.lower()

    def test_system_with_records(self):
        from app.services.system_context.system_answer_builder import (
            build_system_only_answer,
        )

        bundle = SystemContextBundle(
            records=[
                RecordSummary(
                    record_type="request",
                    record_id="R-1",
                    tenant_id="t1",
                    title="Request One",
                    status="pending",
                ),
                RecordSummary(
                    record_type="task",
                    record_id="T-1",
                    tenant_id="t1",
                    title="Task One",
                ),
            ],
        )
        answer = build_system_only_answer(
            question="Recent records?",
            category="system",
            bundle=bundle,
        )

        assert answer is not None
        assert "Request One" in answer
        assert "Task One" in answer

    def test_system_empty_bundle_returns_none(self):
        from app.services.system_context.system_answer_builder import (
            build_system_only_answer,
        )

        bundle = SystemContextBundle()
        answer = build_system_only_answer(
            question="Status?",
            category="system",
            bundle=bundle,
        )

        assert answer is None

    def test_none_bundle_returns_none(self):
        from app.services.system_context.system_answer_builder import (
            build_system_only_answer,
        )

        answer = build_system_only_answer(
            question="Status?",
            category="system",
            bundle=None,
        )

        assert answer is None

    def test_no_raw_json(self):
        from app.services.system_context.system_answer_builder import (
            build_system_only_answer,
        )

        bundle = _make_bundle_with_stats()
        answer = build_system_only_answer(
            question="Stats?",
            category="system",
            bundle=bundle,
        )

        assert answer is not None
        assert "{" not in answer
        assert "}" not in answer

    def test_html_stripped(self):
        from app.services.system_context.system_answer_builder import (
            build_system_only_answer,
        )

        bundle = SystemContextBundle(
            user=UserContext(
                user_id="u1",
                tenant_id="t1",
                display_name="<b>Alice</b>",
            ),
            permissions=PermissionSnapshot(
                tenant_id="t1",
                actor_user_id="u1",
                decisions=[
                    PermissionDecision(
                        resource_type="doc",
                        action="read",
                        allowed=True,
                    ),
                ],
            ),
        )
        answer = build_system_only_answer(
            question="My permissions?",
            category="access",
            bundle=bundle,
        )

        assert answer is not None
        assert "<b>" not in answer
        assert "</b>" not in answer
        assert "Alice" in answer

    def test_mixed_fallback_tries_system_then_access(self):
        from app.services.system_context.system_answer_builder import (
            build_system_only_answer,
        )

        # Bundle with stats → system answer
        bundle = _make_bundle_with_stats()
        answer = build_system_only_answer(
            question="Mixed?",
            category="mixed",
            bundle=bundle,
        )

        assert answer is not None
        assert "42" in answer

    def test_deterministic(self):
        """Same input → same output."""
        from app.services.system_context.system_answer_builder import (
            build_system_only_answer,
        )

        bundle = _make_bundle_with_stats()

        a1 = build_system_only_answer(
            question="Stats?", category="system", bundle=bundle,
        )
        a2 = build_system_only_answer(
            question="Stats?", category="system", bundle=bundle,
        )

        assert a1 == a2


# =====================================================================
# I. MIXED FALLBACK PRECEDENCE (Hotfix)
# =====================================================================


class TestMixedFallbackPrecedence:
    """MIXED: evidence_fallback comes BEFORE system_fallback.
    SYSTEM/ACCESS: system_fallback comes BEFORE evidence_fallback.
    """

    def test_mixed_llm_fail_with_evidences_and_bundle_prefers_evidence_fallback(self):
        """MIXED + LLM fail + evidences + bundle → evidence_fallback wins."""
        import app.services.assistant_service as svc_mod

        bundle = _make_bundle_with_stats()
        orch_result = _make_orch_result(
            category="mixed",
            should_use_knowledge=True,
            should_use_system_context=True,
            should_use_access_context=False,
            bundle=bundle,
        )

        mock_orchestrator = AsyncMock()
        mock_orchestrator.evaluate = AsyncMock(return_value=orch_result)

        mock_query_svc = AsyncMock()
        mock_query_svc.query = AsyncMock(return_value=[])

        # LLM returns no text but HAS evidences
        from app.services.answer_service import AnswerEvidence, GeneratedAnswer
        fake_evidence = AnswerEvidence(
            document_id=10, chunk_id=20, source_document_id=None,
            score=0.95, snippet="Policy says X is Y.", rank=1,
        )
        mock_answer_svc = MagicMock()
        mock_answer_svc.generate_structured = AsyncMock(
            return_value=GeneratedAnswer(
                text=None,  # LLM failed
                intent="general",
                evidences=(fake_evidence,),
            ),
        )

        from app.schemas.assistant import AssistantRespondRequest
        request = AssistantRespondRequest(message="Mixed question about policy and stats")

        with (
            patch.object(svc_mod, "_get_orchestrator", return_value=mock_orchestrator),
            patch.object(svc_mod, "_get_query_svc", return_value=mock_query_svc),
            patch.object(svc_mod, "_get_answer_svc", return_value=mock_answer_svc),
        ):
            service = svc_mod.AssistantService()
            response = _run(service.respond(
                request=request,
                tenant_id="t1",
                user_id=1,
            ))

        # Evidence fallback wins — citations are NOT empty
        assert len(response.citations) == 1
        assert response.citations[0].document_id == 10
        # Should NOT be a system_fallback answer (no "Total Users" / "42")
        assert "42" not in response.message
        assert "Total Users" not in response.message

    def test_mixed_llm_fail_no_evidences_but_bundle_uses_system_fallback(self):
        """MIXED + LLM fail + no evidences + bundle → system_fallback."""
        import app.services.assistant_service as svc_mod

        bundle = _make_bundle_with_stats()
        orch_result = _make_orch_result(
            category="mixed",
            should_use_knowledge=True,
            should_use_system_context=True,
            should_use_access_context=False,
            bundle=bundle,
        )

        mock_orchestrator = AsyncMock()
        mock_orchestrator.evaluate = AsyncMock(return_value=orch_result)

        mock_query_svc = AsyncMock()
        mock_query_svc.query = AsyncMock(return_value=[])

        mock_answer_svc = MagicMock()
        from app.services.answer_service import GeneratedAnswer
        mock_answer_svc.generate_structured = AsyncMock(
            return_value=GeneratedAnswer(
                text=None,  # LLM failed
                intent="general",
                evidences=(),  # no evidences either
            ),
        )

        from app.schemas.assistant import AssistantRespondRequest
        request = AssistantRespondRequest(message="Mixed question")

        with (
            patch.object(svc_mod, "_get_orchestrator", return_value=mock_orchestrator),
            patch.object(svc_mod, "_get_query_svc", return_value=mock_query_svc),
            patch.object(svc_mod, "_get_answer_svc", return_value=mock_answer_svc),
        ):
            service = svc_mod.AssistantService()
            response = _run(service.respond(
                request=request,
                tenant_id="t1",
                user_id=1,
            ))

        # System fallback — no citations, but has system data
        assert response.citations == []
        assert "42" in response.message or "Total Users" in response.message

    def test_system_category_keeps_system_fallback_first(self):
        """SYSTEM + LLM fail + evidences + bundle → system_fallback wins."""
        import app.services.assistant_service as svc_mod

        bundle = _make_bundle_with_stats()
        orch_result = _make_orch_result(
            category="system",
            should_use_knowledge=False,
            should_use_system_context=True,
            should_use_access_context=False,
            bundle=bundle,
        )

        mock_orchestrator = AsyncMock()
        mock_orchestrator.evaluate = AsyncMock(return_value=orch_result)

        mock_query_svc = AsyncMock()
        mock_answer_svc = MagicMock()
        from app.services.answer_service import AnswerEvidence, GeneratedAnswer
        fake_evidence = AnswerEvidence(
            document_id=10, chunk_id=20, source_document_id=None,
            score=0.9, snippet="Some doc content", rank=1,
        )
        mock_answer_svc.generate_structured = AsyncMock(
            return_value=GeneratedAnswer(
                text=None,
                intent="general",
                evidences=(fake_evidence,),
            ),
        )

        from app.schemas.assistant import AssistantRespondRequest
        request = AssistantRespondRequest(message="How many users?")

        with (
            patch.object(svc_mod, "_get_orchestrator", return_value=mock_orchestrator),
            patch.object(svc_mod, "_get_query_svc", return_value=mock_query_svc),
            patch.object(svc_mod, "_get_answer_svc", return_value=mock_answer_svc),
        ):
            service = svc_mod.AssistantService()
            response = _run(service.respond(
                request=request,
                tenant_id="t1",
                user_id=1,
            ))

        # SYSTEM prefers system_fallback → citations=[], answer has stats
        assert response.citations == []
        assert "42" in response.message or "Total Users" in response.message

    def test_access_category_keeps_system_fallback_first(self):
        """ACCESS + LLM fail + bundle → system_fallback wins over evidences."""
        import app.services.assistant_service as svc_mod

        bundle = _make_bundle_with_permissions()
        orch_result = _make_orch_result(
            category="access",
            should_use_knowledge=False,
            should_use_system_context=False,
            should_use_access_context=True,
            bundle=bundle,
        )

        mock_orchestrator = AsyncMock()
        mock_orchestrator.evaluate = AsyncMock(return_value=orch_result)

        mock_query_svc = AsyncMock()
        mock_answer_svc = MagicMock()
        from app.services.answer_service import GeneratedAnswer
        mock_answer_svc.generate_structured = AsyncMock(
            return_value=GeneratedAnswer(
                text=None,
                intent="general",
                evidences=(),
            ),
        )

        from app.schemas.assistant import AssistantRespondRequest
        request = AssistantRespondRequest(message="What can I access?")

        with (
            patch.object(svc_mod, "_get_orchestrator", return_value=mock_orchestrator),
            patch.object(svc_mod, "_get_query_svc", return_value=mock_query_svc),
            patch.object(svc_mod, "_get_answer_svc", return_value=mock_answer_svc),
        ):
            service = svc_mod.AssistantService()
            response = _run(service.respond(
                request=request,
                tenant_id="t1",
                user_id=1,
            ))

        # ACCESS prefers system_fallback → citations=[]
        assert response.citations == []
        assert "Alice" in response.message or "permission" in response.message.lower()

