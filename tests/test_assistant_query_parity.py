"""
Assistant ↔ Query history parity tests.

Covers:
  1. Assistant path passes history to query layer (retrieval branch)
  2. Follow-up queries via assistant have history-aware rewrite parity
  3. Pure SYSTEM/ACCESS paths are NOT affected
  4. Rewrite failure still fails-safe (pipeline continues)
  5. Empty history works without error

Uses asyncio.run() for async tests (no pytest-asyncio dependency).
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.schemas.assistant import (
    AssistantConversationTurn,
    AssistantRespondRequest,
)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_orch_result(
    *,
    category: str,
    should_use_knowledge: bool,
    should_use_system_context: bool,
    should_use_access_context: bool,
    bundle=None,
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


def _stub_service_mocks(*, orch_result=None, query_return=None):
    """Build common mock set for assistant service tests.

    Returns (mock_orchestrator, mock_query_svc, mock_answer_svc).
    """
    from app.services.answer_service import GeneratedAnswer

    mock_orchestrator = AsyncMock()
    if orch_result is not None:
        mock_orchestrator.evaluate = AsyncMock(return_value=orch_result)
    else:
        mock_orchestrator.evaluate = AsyncMock(return_value=_make_orch_result(
            category="knowledge",
            should_use_knowledge=True,
            should_use_system_context=False,
            should_use_access_context=False,
        ))

    mock_query_svc = AsyncMock()
    mock_query_svc.query = AsyncMock(return_value=query_return or [])

    mock_answer_svc = MagicMock()
    mock_answer_svc.generate_structured = AsyncMock(
        return_value=GeneratedAnswer(
            text="Test answer", intent="general", evidences=(),
        ),
    )

    return mock_orchestrator, mock_query_svc, mock_answer_svc


# =====================================================================
# 1. HISTORY PASSED TO QUERY LAYER (retrieval branch)
# =====================================================================


class TestHistoryPassedToQueryLayer:
    """Core parity: assistant path must pass history to query_svc.query()."""

    def test_history_forwarded_to_query_svc(self):
        """Non-empty history is forwarded to query layer in retrieval branch."""
        import app.services.assistant_service as svc_mod

        history = [
            AssistantConversationTurn(role="user", text="chính sách nghỉ phép"),
            AssistantConversationTurn(role="assistant", text="Nghỉ phép tối đa 12 ngày"),
        ]
        request = AssistantRespondRequest(
            message="cái này áp dụng cho ai?",
            history=history,
        )

        mock_orch, mock_query_svc, mock_answer_svc = _stub_service_mocks()

        with (
            patch.object(svc_mod, "_get_orchestrator", return_value=mock_orch),
            patch.object(svc_mod, "_get_query_svc", return_value=mock_query_svc),
            patch.object(svc_mod, "_get_answer_svc", return_value=mock_answer_svc),
        ):
            service = svc_mod.AssistantService()
            _run(service.respond(
                request=request, tenant_id="t1", user_id=1,
            ))

        # Assert: query_svc.query was called with history
        mock_query_svc.query.assert_called_once()
        call_kwargs = mock_query_svc.query.call_args.kwargs
        assert "history" in call_kwargs
        assert len(call_kwargs["history"]) == 2
        assert call_kwargs["history"][0].role == "user"
        assert call_kwargs["history"][0].text == "chính sách nghỉ phép"

    def test_empty_history_forwarded_as_empty_list(self):
        """Empty history → query layer receives [] (not None)."""
        import app.services.assistant_service as svc_mod

        request = AssistantRespondRequest(
            message="What is policy X?",
            history=[],
        )

        mock_orch, mock_query_svc, mock_answer_svc = _stub_service_mocks()

        with (
            patch.object(svc_mod, "_get_orchestrator", return_value=mock_orch),
            patch.object(svc_mod, "_get_query_svc", return_value=mock_query_svc),
            patch.object(svc_mod, "_get_answer_svc", return_value=mock_answer_svc),
        ):
            service = svc_mod.AssistantService()
            _run(service.respond(
                request=request, tenant_id="t1", user_id=1,
            ))

        call_kwargs = mock_query_svc.query.call_args.kwargs
        assert call_kwargs["history"] == []

    def test_no_history_field_defaults_to_empty(self):
        """Request without history field → defaults to [] (schema default)."""
        import app.services.assistant_service as svc_mod

        request = AssistantRespondRequest(message="Simple question")

        mock_orch, mock_query_svc, mock_answer_svc = _stub_service_mocks()

        with (
            patch.object(svc_mod, "_get_orchestrator", return_value=mock_orch),
            patch.object(svc_mod, "_get_query_svc", return_value=mock_query_svc),
            patch.object(svc_mod, "_get_answer_svc", return_value=mock_answer_svc),
        ):
            service = svc_mod.AssistantService()
            _run(service.respond(
                request=request, tenant_id="t1", user_id=1,
            ))

        call_kwargs = mock_query_svc.query.call_args.kwargs
        assert call_kwargs["history"] == []


# =====================================================================
# 2. FOLLOW-UP PARITY: rewrite uses history from assistant path
# =====================================================================


class TestFollowUpParity:
    """History enables follow-up resolution in assistant path (same as query)."""

    def test_mixed_category_also_passes_history(self):
        """MIXED category → retrieval runs → history forwarded."""
        import app.services.assistant_service as svc_mod

        orch_result = _make_orch_result(
            category="mixed",
            should_use_knowledge=True,
            should_use_system_context=True,
            should_use_access_context=False,
        )

        history = [
            AssistantConversationTurn(role="user", text="last question"),
        ]
        request = AssistantRespondRequest(
            message="What about this one?",
            history=history,
        )

        mock_orch, mock_query_svc, mock_answer_svc = _stub_service_mocks(
            orch_result=orch_result,
        )

        with (
            patch.object(svc_mod, "_get_orchestrator", return_value=mock_orch),
            patch.object(svc_mod, "_get_query_svc", return_value=mock_query_svc),
            patch.object(svc_mod, "_get_answer_svc", return_value=mock_answer_svc),
        ):
            service = svc_mod.AssistantService()
            _run(service.respond(
                request=request, tenant_id="t1", user_id=1,
            ))

        call_kwargs = mock_query_svc.query.call_args.kwargs
        assert call_kwargs["history"] == history

    def test_orchestrator_fallback_still_passes_history(self):
        """Orchestrator failure → knowledge fallback → history still forwarded."""
        import app.services.assistant_service as svc_mod

        mock_orch = AsyncMock()
        mock_orch.evaluate = AsyncMock(
            side_effect=RuntimeError("orchestrator error"),
        )

        history = [
            AssistantConversationTurn(role="user", text="previous question"),
        ]
        request = AssistantRespondRequest(
            message="cái này thì sao?",
            history=history,
        )

        mock_query_svc = AsyncMock()
        mock_query_svc.query = AsyncMock(return_value=[])
        _, _, mock_answer_svc = _stub_service_mocks()

        with (
            patch.object(svc_mod, "_get_orchestrator", return_value=mock_orch),
            patch.object(svc_mod, "_get_query_svc", return_value=mock_query_svc),
            patch.object(svc_mod, "_get_answer_svc", return_value=mock_answer_svc),
        ):
            service = svc_mod.AssistantService()
            _run(service.respond(
                request=request, tenant_id="t1", user_id=1,
            ))

        # Despite orchestrator failure, retrieval still runs with history
        mock_query_svc.query.assert_called_once()
        call_kwargs = mock_query_svc.query.call_args.kwargs
        assert call_kwargs["history"] == history


# =====================================================================
# 3. SYSTEM/ACCESS PATHS NOT AFFECTED
# =====================================================================


class TestSystemAccessUnaffected:
    """Pure SYSTEM/ACCESS paths skip retrieval — history does not leak."""

    def test_system_category_query_not_called(self):
        """SYSTEM → retrieval skipped, no query_svc.query() call at all."""
        import app.services.assistant_service as svc_mod

        orch_result = _make_orch_result(
            category="system",
            should_use_knowledge=False,
            should_use_system_context=True,
            should_use_access_context=False,
        )

        history = [
            AssistantConversationTurn(role="user", text="previous q"),
        ]
        request = AssistantRespondRequest(
            message="How many users?",
            history=history,
        )

        mock_orch, mock_query_svc, mock_answer_svc = _stub_service_mocks(
            orch_result=orch_result,
        )

        with (
            patch.object(svc_mod, "_get_orchestrator", return_value=mock_orch),
            patch.object(svc_mod, "_get_query_svc", return_value=mock_query_svc),
            patch.object(svc_mod, "_get_answer_svc", return_value=mock_answer_svc),
        ):
            service = svc_mod.AssistantService()
            _run(service.respond(
                request=request, tenant_id="t1", user_id=1,
            ))

        # Retrieval was NOT called (SYSTEM skips retrieval entirely)
        mock_query_svc.query.assert_not_called()

    def test_access_category_query_not_called(self):
        """ACCESS → retrieval skipped, no query_svc.query() call at all."""
        import app.services.assistant_service as svc_mod

        orch_result = _make_orch_result(
            category="access",
            should_use_knowledge=False,
            should_use_system_context=False,
            should_use_access_context=True,
        )

        history = [
            AssistantConversationTurn(role="user", text="prev"),
        ]
        request = AssistantRespondRequest(
            message="What can I access?",
            history=history,
        )

        mock_orch, mock_query_svc, mock_answer_svc = _stub_service_mocks(
            orch_result=orch_result,
        )

        with (
            patch.object(svc_mod, "_get_orchestrator", return_value=mock_orch),
            patch.object(svc_mod, "_get_query_svc", return_value=mock_query_svc),
            patch.object(svc_mod, "_get_answer_svc", return_value=mock_answer_svc),
        ):
            service = svc_mod.AssistantService()
            _run(service.respond(
                request=request, tenant_id="t1", user_id=1,
            ))

        mock_query_svc.query.assert_not_called()


# =====================================================================
# 4. REWRITE FAILURE DOES NOT BREAK PIPELINE
# =====================================================================


class TestRewriteFailSafe:
    """If query rewrite service fails, pipeline continues (fail-open)."""

    def test_query_svc_receives_history_even_if_rewrite_would_fail(self):
        """History is passed at the QueryService level regardless of
        whether QueryRewriteService handles it successfully.
        The fail-open is inside QueryService — not our concern here.
        We only verify assistant_service passes it correctly.
        """
        import app.services.assistant_service as svc_mod

        history = [
            AssistantConversationTurn(role="user", text="bad data"),
        ]
        request = AssistantRespondRequest(
            message="follow up question",
            history=history,
        )

        mock_orch, mock_query_svc, mock_answer_svc = _stub_service_mocks()

        with (
            patch.object(svc_mod, "_get_orchestrator", return_value=mock_orch),
            patch.object(svc_mod, "_get_query_svc", return_value=mock_query_svc),
            patch.object(svc_mod, "_get_answer_svc", return_value=mock_answer_svc),
        ):
            service = svc_mod.AssistantService()
            response = _run(service.respond(
                request=request, tenant_id="t1", user_id=1,
            ))

        # Pipeline completes with a response
        assert response.message is not None and len(response.message) > 0

        # History was passed
        call_kwargs = mock_query_svc.query.call_args.kwargs
        assert call_kwargs["history"] == history


# =====================================================================
# 5. HISTORY ALSO PASSED TO ANSWER SERVICE (already worked, regression)
# =====================================================================


class TestAnswerServiceHistoryRegression:
    """Verify answer_svc.generate_structured still receives history
    (this already worked — ensure the change doesn't break it).
    """

    def test_answer_svc_receives_history(self):
        import app.services.assistant_service as svc_mod

        history = [
            AssistantConversationTurn(role="user", text="q1"),
            AssistantConversationTurn(role="assistant", text="a1"),
        ]
        request = AssistantRespondRequest(
            message="follow up q2",
            history=history,
        )

        mock_orch, mock_query_svc, mock_answer_svc = _stub_service_mocks()

        with (
            patch.object(svc_mod, "_get_orchestrator", return_value=mock_orch),
            patch.object(svc_mod, "_get_query_svc", return_value=mock_query_svc),
            patch.object(svc_mod, "_get_answer_svc", return_value=mock_answer_svc),
        ):
            service = svc_mod.AssistantService()
            _run(service.respond(
                request=request, tenant_id="t1", user_id=1,
            ))

        # answer_svc also receives history
        answer_kwargs = mock_answer_svc.generate_structured.call_args.kwargs
        assert answer_kwargs["history"] == history
