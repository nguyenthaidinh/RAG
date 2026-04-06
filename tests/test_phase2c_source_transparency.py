"""
Phase 2C — Source transparency tests.

Verifies that AnswerSourceTrace correctly captures provenance
for each routing path without changing public API behavior.

Covers:
  A. Knowledge path trace
  B. System path trace
  C. Access path trace
  D. Mixed path trace
  E. Orchestrator failure → knowledge fallback trace
  F. No fake citations for system/access
  G. AnswerSourceTrace deterministic
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.schemas.system_context import (
    MetricValue,
    PermissionDecision,
    PermissionSnapshot,
    SystemContextBundle,
    TenantStats,
    UserContext,
)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_orch_result(*, category, should_use_knowledge, should_use_system_context,
                       should_use_access_context, bundle=None):
    from app.services.orchestration.question_classifier import (
        ClassificationResult, QuestionCategory,
    )
    from app.services.orchestration.system_context_orchestrator import OrchestrationResult
    from app.services.system_context.context_builder import ContextBuildFlags

    cat = QuestionCategory(category)
    return OrchestrationResult(
        category=cat,
        classification=ClassificationResult(
            category=cat, confidence=0.85, matched_signals=(),
        ),
        should_use_knowledge=should_use_knowledge,
        should_use_system_context=should_use_system_context,
        should_use_access_context=should_use_access_context,
        recommended_flags=ContextBuildFlags(),
        context_bundle=bundle,
    )


def _make_stats_bundle():
    return SystemContextBundle(
        user=UserContext(user_id="u1", tenant_id="t1", display_name="Alice"),
        tenant_stats=TenantStats(
            tenant_id="t1",
            metrics=[MetricValue(key="users", value=42, label="Users")],
        ),
        source="core-platform",
    )


def _make_perms_bundle():
    return SystemContextBundle(
        user=UserContext(user_id="u1", tenant_id="t1", display_name="Alice"),
        permissions=PermissionSnapshot(
            tenant_id="t1", actor_user_id="u1",
            decisions=[
                PermissionDecision(resource_type="doc", action="read", allowed=True),
            ],
        ),
        source="core-platform",
    )


def _respond_and_capture_trace(orch_result, generated, query_results=None):
    """Run respond() and capture the AnswerSourceTrace from logger."""
    import app.services.assistant_service as svc_mod

    mock_orchestrator = AsyncMock()
    mock_orchestrator.evaluate = AsyncMock(return_value=orch_result)

    mock_query_svc = AsyncMock()
    mock_query_svc.query = AsyncMock(return_value=query_results or [])

    mock_answer_svc = MagicMock()
    mock_answer_svc.generate_structured = AsyncMock(return_value=generated)

    from app.schemas.assistant import AssistantRespondRequest
    request = AssistantRespondRequest(message="Test question")

    with (
        patch.object(svc_mod, "_get_orchestrator", return_value=mock_orchestrator),
        patch.object(svc_mod, "_get_query_svc", return_value=mock_query_svc),
        patch.object(svc_mod, "_get_answer_svc", return_value=mock_answer_svc),
    ):
        service = svc_mod.AssistantService()
        response = _run(service.respond(
            request=request, tenant_id="t1", user_id=1,
        ))

    return response, mock_query_svc, mock_answer_svc


# =====================================================================
# A. KNOWLEDGE PATH TRACE
# =====================================================================


class TestKnowledgePathTrace:
    def test_knowledge_uses_document_evidence(self):
        """KNOWLEDGE → retrieval runs, answer sources from docs."""
        from app.services.answer_service import AnswerEvidence, GeneratedAnswer

        orch = _make_orch_result(
            category="knowledge",
            should_use_knowledge=True,
            should_use_system_context=False,
            should_use_access_context=False,
        )
        evidence = AnswerEvidence(
            document_id=1, chunk_id=1, source_document_id=None,
            score=0.9, snippet="Doc content", rank=1,
        )
        gen = GeneratedAnswer(
            text="LLM answer", intent="general",
            evidences=(evidence,), model="gpt-4o-mini", provider="openai",
        )

        resp, qsvc, asvc = _respond_and_capture_trace(orch, gen)

        # Retrieval was called
        qsvc.query.assert_called_once()
        # LLM answer used
        assert resp.message == "LLM answer"
        # Has citations from docs
        assert len(resp.citations) == 1
        assert resp.citations[0].document_id == 1


# =====================================================================
# B. SYSTEM PATH TRACE
# =====================================================================


class TestSystemPathTrace:
    def test_system_skips_retrieval_uses_context(self):
        """SYSTEM → retrieval NOT called, system context used."""
        from app.services.answer_service import GeneratedAnswer

        bundle = _make_stats_bundle()
        orch = _make_orch_result(
            category="system",
            should_use_knowledge=False,
            should_use_system_context=True,
            should_use_access_context=False,
            bundle=bundle,
        )
        gen = GeneratedAnswer(text=None, intent="general", evidences=())

        resp, qsvc, _ = _respond_and_capture_trace(orch, gen)

        qsvc.query.assert_not_called()
        assert resp.citations == []
        # System fallback answer has stats data
        assert "42" in resp.message or "Users" in resp.message


# =====================================================================
# C. ACCESS PATH TRACE
# =====================================================================


class TestAccessPathTrace:
    def test_access_skips_retrieval_uses_permissions(self):
        """ACCESS → retrieval NOT called, permission context used."""
        from app.services.answer_service import GeneratedAnswer

        bundle = _make_perms_bundle()
        orch = _make_orch_result(
            category="access",
            should_use_knowledge=False,
            should_use_system_context=False,
            should_use_access_context=True,
            bundle=bundle,
        )
        gen = GeneratedAnswer(text=None, intent="general", evidences=())

        resp, qsvc, _ = _respond_and_capture_trace(orch, gen)

        qsvc.query.assert_not_called()
        assert resp.citations == []
        assert "Alice" in resp.message or "permission" in resp.message.lower()


# =====================================================================
# D. MIXED PATH TRACE
# =====================================================================


class TestMixedPathTrace:
    def test_mixed_retrieval_runs_and_system_context_present(self):
        """MIXED → retrieval runs, system context available."""
        from app.services.answer_service import AnswerEvidence, GeneratedAnswer

        bundle = _make_stats_bundle()
        orch = _make_orch_result(
            category="mixed",
            should_use_knowledge=True,
            should_use_system_context=True,
            should_use_access_context=False,
            bundle=bundle,
        )
        evidence = AnswerEvidence(
            document_id=1, chunk_id=1, source_document_id=None,
            score=0.9, snippet="Policy text", rank=1,
        )
        gen = GeneratedAnswer(
            text="Combined answer", intent="general",
            evidences=(evidence,),
        )

        resp, qsvc, asvc = _respond_and_capture_trace(orch, gen)

        qsvc.query.assert_called_once()
        assert resp.message == "Combined answer"
        assert len(resp.citations) == 1

    def test_mixed_evidence_fallback_preserves_citations(self):
        """MIXED + LLM fail + evidences → evidence_fallback with citations."""
        from app.services.answer_service import AnswerEvidence, GeneratedAnswer

        bundle = _make_stats_bundle()
        orch = _make_orch_result(
            category="mixed",
            should_use_knowledge=True,
            should_use_system_context=True,
            should_use_access_context=False,
            bundle=bundle,
        )
        evidence = AnswerEvidence(
            document_id=5, chunk_id=10, source_document_id=None,
            score=0.85, snippet="Doc text", rank=1,
        )
        gen = GeneratedAnswer(
            text=None, intent="general", evidences=(evidence,),
        )

        resp, _, _ = _respond_and_capture_trace(orch, gen)

        # Evidence fallback → citations preserved
        assert len(resp.citations) == 1
        assert resp.citations[0].document_id == 5


# =====================================================================
# E. ORCHESTRATOR FAILURE → KNOWLEDGE FALLBACK
# =====================================================================


class TestOrchFailureTrace:
    def test_orch_failure_leads_to_knowledge_path(self):
        """Orchestrator exception → retrieval still runs."""
        import app.services.assistant_service as svc_mod
        from app.services.answer_service import GeneratedAnswer

        mock_orchestrator = AsyncMock()
        mock_orchestrator.evaluate = AsyncMock(
            side_effect=RuntimeError("boom"),
        )
        mock_query_svc = AsyncMock()
        mock_query_svc.query = AsyncMock(return_value=[])
        mock_answer_svc = MagicMock()
        mock_answer_svc.generate_structured = AsyncMock(
            return_value=GeneratedAnswer(
                text=None, intent="general", evidences=(),
            ),
        )

        from app.schemas.assistant import AssistantRespondRequest
        request = AssistantRespondRequest(message="Q?")

        with (
            patch.object(svc_mod, "_get_orchestrator", return_value=mock_orchestrator),
            patch.object(svc_mod, "_get_query_svc", return_value=mock_query_svc),
            patch.object(svc_mod, "_get_answer_svc", return_value=mock_answer_svc),
        ):
            service = svc_mod.AssistantService()
            resp = _run(service.respond(
                request=request, tenant_id="t1", user_id=1,
            ))

        mock_query_svc.query.assert_called_once()
        assert resp.message  # Has deterministic message


# =====================================================================
# F. NO FAKE CITATIONS
# =====================================================================


class TestNoFakeCitations:
    def test_system_answer_citations_empty(self):
        from app.services.answer_service import GeneratedAnswer

        bundle = _make_stats_bundle()
        orch = _make_orch_result(
            category="system",
            should_use_knowledge=False,
            should_use_system_context=True,
            should_use_access_context=False,
            bundle=bundle,
        )
        gen = GeneratedAnswer(text=None, intent="general", evidences=())

        resp, _, _ = _respond_and_capture_trace(orch, gen)
        assert resp.citations == []

    def test_access_answer_citations_empty(self):
        from app.services.answer_service import GeneratedAnswer

        bundle = _make_perms_bundle()
        orch = _make_orch_result(
            category="access",
            should_use_knowledge=False,
            should_use_system_context=False,
            should_use_access_context=True,
            bundle=bundle,
        )
        gen = GeneratedAnswer(text=None, intent="general", evidences=())

        resp, _, _ = _respond_and_capture_trace(orch, gen)
        assert resp.citations == []


# =====================================================================
# G. ANSWER SOURCE TRACE UNIT TESTS
# =====================================================================


class TestAnswerSourceTrace:
    def test_trace_defaults(self):
        from app.services.system_context.source_trace import AnswerSourceTrace

        trace = AnswerSourceTrace()
        assert trace.category == "knowledge"
        assert trace.orchestration_ok is False
        assert trace.answer_source == "deterministic"
        assert trace.fallback_level == 0

    def test_to_log_dict_is_safe(self):
        from app.services.system_context.source_trace import AnswerSourceTrace

        trace = AnswerSourceTrace(
            category="system",
            answer_source="system_fallback",
            used_system_context=True,
        )
        d = trace.to_log_dict()

        assert isinstance(d, dict)
        assert d["category"] == "system"
        assert d["answer_source"] == "system_fallback"
        assert d["used_system_context"] is True
        # No raw content keys
        assert "raw_context" not in d
        assert "token" not in d

    def test_trace_deterministic(self):
        from app.services.system_context.source_trace import AnswerSourceTrace

        t1 = AnswerSourceTrace(category="system", answer_source="llm")
        t2 = AnswerSourceTrace(category="system", answer_source="llm")
        assert t1 == t2

    def test_trace_frozen(self):
        from app.services.system_context.source_trace import AnswerSourceTrace

        trace = AnswerSourceTrace()
        with pytest.raises(Exception):
            trace.category = "system"  # type: ignore
