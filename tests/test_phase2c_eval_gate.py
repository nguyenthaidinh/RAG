"""
Phase 2C — Eval gate / staging verification tests.

Mini eval gate: deterministic tests that lock expected behavior
for each question category without requiring an LLM.

For each category verifies:
  - Route category expected
  - Retrieval expected yes/no
  - System context expected yes/no
  - Fallback expected if LLM fail
  - Citations expected yes/no

All mocked at service level — no external calls.
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
    TenantStats,
    UserContext,
    WorkflowSummary,
)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_orch_result(*, category, should_use_knowledge,
                       should_use_system_context, should_use_access_context,
                       bundle=None):
    from app.services.orchestration.question_classifier import (
        ClassificationResult, QuestionCategory,
    )
    from app.services.orchestration.system_context_orchestrator import OrchestrationResult
    from app.services.system_context.context_builder import ContextBuildFlags

    cat = QuestionCategory(category)
    return OrchestrationResult(
        category=cat,
        classification=ClassificationResult(
            category=cat, confidence=0.9, matched_signals=(),
        ),
        should_use_knowledge=should_use_knowledge,
        should_use_system_context=should_use_system_context,
        should_use_access_context=should_use_access_context,
        recommended_flags=ContextBuildFlags(),
        context_bundle=bundle,
    )


def _full_bundle():
    return SystemContextBundle(
        user=UserContext(
            user_id="u1", tenant_id="t1",
            display_name="Alice", role="admin",
        ),
        tenant_stats=TenantStats(
            tenant_id="t1",
            metrics=[
                MetricValue(key="users", value=42, label="Total Users"),
                MetricValue(key="docs", value=100, label="Docs"),
            ],
        ),
        permissions=PermissionSnapshot(
            tenant_id="t1", actor_user_id="u1",
            decisions=[
                PermissionDecision(resource_type="doc", action="read", allowed=True),
                PermissionDecision(resource_type="user", action="manage", allowed=False),
            ],
        ),
        records=[
            RecordSummary(
                record_type="request", record_id="R-1",
                tenant_id="t1", title="Request One", status="pending",
            ),
        ],
        workflows=[
            WorkflowSummary(
                workflow_type="approval", tenant_id="t1",
                total=50, pending_count=10, completed_count=40,
            ),
        ],
        source="core-platform",
    )


def _eval_respond(orch_result, generated):
    """Run respond and return (response, query_svc_mock, answer_svc_mock)."""
    import app.services.assistant_service as svc_mod

    mock_orch = AsyncMock()
    mock_orch.evaluate = AsyncMock(return_value=orch_result)

    mock_query = AsyncMock()
    mock_query.query = AsyncMock(return_value=[])

    mock_answer = MagicMock()
    mock_answer.generate_structured = AsyncMock(return_value=generated)

    from app.schemas.assistant import AssistantRespondRequest
    req = AssistantRespondRequest(message="Eval question")

    with (
        patch.object(svc_mod, "_get_orchestrator", return_value=mock_orch),
        patch.object(svc_mod, "_get_query_svc", return_value=mock_query),
        patch.object(svc_mod, "_get_answer_svc", return_value=mock_answer),
    ):
        service = svc_mod.AssistantService()
        resp = _run(service.respond(
            request=req, tenant_id="t1", user_id=1,
        ))

    return resp, mock_query, mock_answer


# =====================================================================
# EVAL GATE: KNOWLEDGE
# =====================================================================


class TestEvalKnowledge:
    """Knowledge question: retrieval yes, system no, LLM answer used."""

    def test_llm_success(self):
        from app.services.answer_service import AnswerEvidence, GeneratedAnswer

        orch = _make_orch_result(
            category="knowledge",
            should_use_knowledge=True,
            should_use_system_context=False,
            should_use_access_context=False,
        )
        ev = AnswerEvidence(
            document_id=1, chunk_id=1, source_document_id=None,
            score=0.9, snippet="The policy states X.", rank=1,
        )
        gen = GeneratedAnswer(
            text="Answer from policy.", intent="general",
            evidences=(ev,),
        )

        resp, qsvc, _ = _eval_respond(orch, gen)

        # Retrieval ran
        qsvc.query.assert_called_once()
        # LLM answer used
        assert resp.message == "Answer from policy."
        # Has citations
        assert len(resp.citations) == 1
        assert resp.citations[0].document_id == 1

    def test_llm_fail_fallback(self):
        from app.services.answer_service import AnswerEvidence, GeneratedAnswer

        orch = _make_orch_result(
            category="knowledge",
            should_use_knowledge=True,
            should_use_system_context=False,
            should_use_access_context=False,
        )
        ev = AnswerEvidence(
            document_id=1, chunk_id=1, source_document_id=None,
            score=0.8, snippet="Content.", rank=1,
        )
        gen = GeneratedAnswer(
            text=None, intent="general", evidences=(ev,),
        )

        resp, qsvc, _ = _eval_respond(orch, gen)

        qsvc.query.assert_called_once()
        # Evidence fallback
        assert len(resp.citations) == 1
        assert "relevant content" in resp.message.lower()

    def test_llm_fail_no_results(self):
        from app.services.answer_service import GeneratedAnswer

        orch = _make_orch_result(
            category="knowledge",
            should_use_knowledge=True,
            should_use_system_context=False,
            should_use_access_context=False,
        )
        gen = GeneratedAnswer(text=None, intent="general", evidences=())

        resp, qsvc, _ = _eval_respond(orch, gen)

        qsvc.query.assert_called_once()
        # Deterministic fallback
        assert resp.citations == []
        assert "could not find" in resp.message.lower()


# =====================================================================
# EVAL GATE: SYSTEM
# =====================================================================


class TestEvalSystem:
    """System question: retrieval no, system yes, fallback to stats."""

    def test_llm_success(self):
        from app.services.answer_service import GeneratedAnswer

        bundle = _full_bundle()
        orch = _make_orch_result(
            category="system",
            should_use_knowledge=False,
            should_use_system_context=True,
            should_use_access_context=False,
            bundle=bundle,
        )
        gen = GeneratedAnswer(
            text="There are 42 users.", intent="general", evidences=(),
        )

        resp, qsvc, _ = _eval_respond(orch, gen)

        qsvc.query.assert_not_called()
        assert resp.message == "There are 42 users."
        assert resp.citations == []  # System answers: no citations

    def test_llm_fail_system_fallback(self):
        from app.services.answer_service import GeneratedAnswer

        bundle = _full_bundle()
        orch = _make_orch_result(
            category="system",
            should_use_knowledge=False,
            should_use_system_context=True,
            should_use_access_context=False,
            bundle=bundle,
        )
        gen = GeneratedAnswer(text=None, intent="general", evidences=())

        resp, qsvc, _ = _eval_respond(orch, gen)

        qsvc.query.assert_not_called()
        assert resp.citations == []
        # System fallback contains stats
        assert "42" in resp.message or "Users" in resp.message

    def test_no_bundle_deterministic(self):
        from app.services.answer_service import GeneratedAnswer

        orch = _make_orch_result(
            category="system",
            should_use_knowledge=False,
            should_use_system_context=True,
            should_use_access_context=False,
            bundle=None,
        )
        gen = GeneratedAnswer(text=None, intent="general", evidences=())

        resp, qsvc, _ = _eval_respond(orch, gen)

        qsvc.query.assert_not_called()
        assert "system information" in resp.message.lower() or "administrator" in resp.message.lower()


# =====================================================================
# EVAL GATE: ACCESS
# =====================================================================


class TestEvalAccess:
    """Access question: retrieval no, access yes, fallback to permissions."""

    def test_llm_success(self):
        from app.services.answer_service import GeneratedAnswer

        bundle = _full_bundle()
        orch = _make_orch_result(
            category="access",
            should_use_knowledge=False,
            should_use_system_context=False,
            should_use_access_context=True,
            bundle=bundle,
        )
        gen = GeneratedAnswer(
            text="You can read docs but cannot manage users.",
            intent="general", evidences=(),
        )

        resp, qsvc, _ = _eval_respond(orch, gen)

        qsvc.query.assert_not_called()
        assert "manage users" in resp.message
        assert resp.citations == []

    def test_llm_fail_access_fallback(self):
        from app.services.answer_service import GeneratedAnswer

        bundle = _full_bundle()
        orch = _make_orch_result(
            category="access",
            should_use_knowledge=False,
            should_use_system_context=False,
            should_use_access_context=True,
            bundle=bundle,
        )
        gen = GeneratedAnswer(text=None, intent="general", evidences=())

        resp, qsvc, _ = _eval_respond(orch, gen)

        qsvc.query.assert_not_called()
        assert resp.citations == []
        assert "Alice" in resp.message or "Allowed" in resp.message or "permission" in resp.message.lower()


# =====================================================================
# EVAL GATE: MIXED
# =====================================================================


class TestEvalMixed:
    """Mixed question: retrieval yes, system yes, evidence-first fallback."""

    def test_llm_success(self):
        from app.services.answer_service import AnswerEvidence, GeneratedAnswer

        bundle = _full_bundle()
        orch = _make_orch_result(
            category="mixed",
            should_use_knowledge=True,
            should_use_system_context=True,
            should_use_access_context=False,
            bundle=bundle,
        )
        ev = AnswerEvidence(
            document_id=2, chunk_id=3, source_document_id=None,
            score=0.88, snippet="Policy plus stats.", rank=1,
        )
        gen = GeneratedAnswer(
            text="Policy says X and there are 42 users.",
            intent="general", evidences=(ev,),
        )

        resp, qsvc, _ = _eval_respond(orch, gen)

        qsvc.query.assert_called_once()
        assert "42 users" in resp.message
        assert len(resp.citations) == 1

    def test_llm_fail_with_evidences_prefers_evidence(self):
        """MIXED + LLM fail + evidences → evidence fallback (not system)."""
        from app.services.answer_service import AnswerEvidence, GeneratedAnswer

        bundle = _full_bundle()
        orch = _make_orch_result(
            category="mixed",
            should_use_knowledge=True,
            should_use_system_context=True,
            should_use_access_context=False,
            bundle=bundle,
        )
        ev = AnswerEvidence(
            document_id=7, chunk_id=14, source_document_id=None,
            score=0.9, snippet="Doc evidence.", rank=1,
        )
        gen = GeneratedAnswer(text=None, intent="general", evidences=(ev,))

        resp, qsvc, _ = _eval_respond(orch, gen)

        qsvc.query.assert_called_once()
        # Evidence fallback: has citations
        assert len(resp.citations) == 1
        assert resp.citations[0].document_id == 7
        # Not a system_fallback answer
        assert "42" not in resp.message

    def test_llm_fail_no_evidences_system_fallback(self):
        """MIXED + LLM fail + no evidences → system fallback."""
        from app.services.answer_service import GeneratedAnswer

        bundle = _full_bundle()
        orch = _make_orch_result(
            category="mixed",
            should_use_knowledge=True,
            should_use_system_context=True,
            should_use_access_context=False,
            bundle=bundle,
        )
        gen = GeneratedAnswer(text=None, intent="general", evidences=())

        resp, qsvc, _ = _eval_respond(orch, gen)

        qsvc.query.assert_called_once()
        # System fallback
        assert resp.citations == []
        assert "42" in resp.message or "Users" in resp.message


# =====================================================================
# EVAL GATE: CROSS-CATEGORY INVARIANTS
# =====================================================================


class TestCrossCategoryInvariants:
    """Invariants that must hold across all categories."""

    def test_all_responses_have_message(self):
        """Every category must return a non-empty message."""
        from app.services.answer_service import GeneratedAnswer

        bundle = _full_bundle()
        categories = [
            ("knowledge", True, False, False, None),
            ("system", False, True, False, bundle),
            ("access", False, False, True, bundle),
            ("mixed", True, True, False, bundle),
        ]

        for cat, know, sys, acc, b in categories:
            orch = _make_orch_result(
                category=cat,
                should_use_knowledge=know,
                should_use_system_context=sys,
                should_use_access_context=acc,
                bundle=b,
            )
            gen = GeneratedAnswer(text=None, intent="general", evidences=())
            resp, _, _ = _eval_respond(orch, gen)

            assert resp.message, f"Empty message for category={cat}"
            assert len(resp.message) > 5, f"Message too short for category={cat}"

    def test_system_access_never_have_doc_citations(self):
        """SYSTEM and ACCESS with LLM fail must not have doc citations."""
        from app.services.answer_service import GeneratedAnswer

        bundle = _full_bundle()
        for cat, sys, acc in [("system", True, False), ("access", False, True)]:
            orch = _make_orch_result(
                category=cat,
                should_use_knowledge=False,
                should_use_system_context=sys,
                should_use_access_context=acc,
                bundle=bundle,
            )
            gen = GeneratedAnswer(text=None, intent="general", evidences=())
            resp, _, _ = _eval_respond(orch, gen)

            assert resp.citations == [], f"Fake citations for category={cat}"
