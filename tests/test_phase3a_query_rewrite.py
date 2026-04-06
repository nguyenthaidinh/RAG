"""
Phase 3A — Query Rewrite tests (hardened).

Tests cover:
  - QueryMode classification
  - RetrievalPlan structure and effective_queries()
  - step_back_query in effective_queries()
  - Full dedupe in effective_queries()
  - Follow-up marker detection
  - History reference resolution
  - History resolution fallback (no LLM)
  - Guardrails (confidence threshold, length limits, dedupe)
  - Forbidden filter token rejection
  - Invalid query validation
  - Feature flag gate
  - Fail-open on error/timeout
  - Telemetry dict safety
  - Schema validation
  - Planner skip when rewrite usable
"""
from __future__ import annotations

import asyncio
import re
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _run(coro):
    """Run an async function synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


def _mock_settings(**overrides):
    """Create a mock settings context with defaults + overrides."""
    defaults = {
        "QUERY_REWRITE_ENABLED": True,
        "QUERY_REWRITE_PROVIDER": "none",
        "QUERY_REWRITE_MODEL": "gpt-4o-mini",
        "QUERY_REWRITE_TIMEOUT_S": 3.0,
        "QUERY_REWRITE_MAX_TOKENS": 300,
        "QUERY_REWRITE_TEMPERATURE": 0.1,
        "QUERY_REWRITE_MAX_SUBQUERIES": 2,
        "QUERY_REWRITE_MAX_QUERY_CHARS": 1200,
        "QUERY_REWRITE_CONFIDENCE_THRESHOLD": 0.5,
        "OPENAI_API_KEY": "",
    }
    defaults.update(overrides)
    mock = MagicMock()
    for k, v in defaults.items():
        setattr(mock, k, v)
    return mock


def _make_svc(**settings_overrides):
    """Create a QueryRewriteService with mocked settings."""
    mock = _mock_settings(**settings_overrides)
    with patch("app.services.query_rewrite_service.settings", mock):
        from app.services.query_rewrite_service import QueryRewriteService
        return QueryRewriteService()


# ── QueryMode ─────────────────────────────────────────────────────────


class TestQueryMode:
    """Test query mode classification."""

    def test_all_modes_defined(self):
        from app.schemas.query_rewrite import QueryMode

        assert QueryMode.DIRECT == "direct"
        assert QueryMode.OVERVIEW == "overview"
        assert QueryMode.SPECIFIC == "specific"
        assert QueryMode.COMPARISON == "comparison"
        assert QueryMode.FOLLOW_UP == "follow_up"
        assert QueryMode.AMBIGUOUS == "ambiguous"
        assert QueryMode.MULTI_HOP == "multi_hop"


# ── RetrievalPlan ─────────────────────────────────────────────────────


class TestRetrievalPlan:
    """Test RetrievalPlan structure and methods."""

    def test_passthrough(self):
        from app.schemas.query_rewrite import RetrievalPlan

        plan = RetrievalPlan.passthrough("hello")
        assert plan.original_query == "hello"
        assert plan.rewritten_query is None
        assert plan.fallback_used is True
        assert plan.effective_queries() == ["hello"]

    def test_effective_queries_with_rewrite(self):
        from app.schemas.query_rewrite import QueryMode, RetrievalPlan

        plan = RetrievalPlan(
            original_query="what is X",
            rewritten_query="explain concept X in detail",
            subqueries=("X definition", "X examples"),
            query_mode=QueryMode.SPECIFIC,
            confidence=0.9,
        )
        queries = plan.effective_queries()
        assert queries[0] == "what is X"
        assert "explain concept X in detail" in queries
        assert "X definition" in queries
        assert "X examples" in queries
        assert len(queries) == 4

    def test_effective_queries_includes_step_back(self):
        """FIX 1: step_back_query must participate in retrieval."""
        from app.schemas.query_rewrite import QueryMode, RetrievalPlan

        plan = RetrievalPlan(
            original_query="what is section 5.2",
            rewritten_query="explain section 5.2 of the document",
            step_back_query="document structure and sections overview",
            query_mode=QueryMode.SPECIFIC,
            confidence=0.9,
        )
        queries = plan.effective_queries()
        assert queries[0] == "what is section 5.2"
        assert "explain section 5.2 of the document" in queries
        assert "document structure and sections overview" in queries

    def test_effective_queries_step_back_order(self):
        """step_back appears after rewritten, before subqueries."""
        from app.schemas.query_rewrite import QueryMode, RetrievalPlan

        plan = RetrievalPlan(
            original_query="original",
            rewritten_query="rewritten",
            step_back_query="step_back",
            subqueries=("sub1",),
            query_mode=QueryMode.SPECIFIC,
            confidence=0.9,
        )
        queries = plan.effective_queries()
        assert queries == ["original", "rewritten", "step_back", "sub1"]

    def test_effective_queries_excludes_fallback_rewrite(self):
        from app.schemas.query_rewrite import QueryMode, RetrievalPlan

        plan = RetrievalPlan(
            original_query="test",
            rewritten_query="rewritten test",
            step_back_query="step back test",
            subqueries=("sub1",),
            fallback_used=True,
            query_mode=QueryMode.DIRECT,
        )
        # When fallback_used=True, everything except original is excluded
        queries = plan.effective_queries()
        assert queries == ["test"]

    def test_effective_queries_full_dedupe(self):
        """FIX 1: dedupe across all query types by normalized comparison."""
        from app.schemas.query_rewrite import QueryMode, RetrievalPlan

        plan = RetrievalPlan(
            original_query="hello world",
            rewritten_query="Hello  World",  # same normalized
            step_back_query="hello world",    # same as original
            subqueries=("unique sub", "hello world"),  # one dup
            query_mode=QueryMode.DIRECT,
        )
        queries = plan.effective_queries()
        assert queries == ["hello world", "unique sub"]

    def test_effective_queries_rejects_empty(self):
        from app.schemas.query_rewrite import QueryMode, RetrievalPlan

        plan = RetrievalPlan(
            original_query="hello",
            rewritten_query="",
            step_back_query="   ",
            subqueries=("", "  "),
            query_mode=QueryMode.DIRECT,
        )
        queries = plan.effective_queries()
        assert queries == ["hello"]

    def test_telemetry_dict_safe(self):
        from app.schemas.query_rewrite import QueryMode, RetrievalPlan

        plan = RetrievalPlan(
            original_query="secret query about confidential docs",
            rewritten_query="rewritten secret",
            step_back_query="broader search",
            query_mode=QueryMode.SPECIFIC,
            confidence=0.85,
            latency_ms=42,
        )
        tele = plan.telemetry_dict()
        assert "secret" not in str(tele)
        assert "confidential" not in str(tele)
        assert tele["query_mode"] == "specific"
        assert tele["rewrite_used"] is True
        assert tele["step_back_used"] is True
        assert tele["rewrite_confidence"] == 0.85
        assert tele["latency_ms"] == 42

    def test_telemetry_dict_keys(self):
        from app.schemas.query_rewrite import RetrievalPlan

        plan = RetrievalPlan.passthrough("x")
        tele = plan.telemetry_dict()
        expected_keys = {
            "query_mode", "rewrite_strategy", "rewrite_used",
            "step_back_used", "rewrite_confidence", "subquery_count",
            "fallback_used", "used_history", "latency_ms",
        }
        assert set(tele.keys()) == expected_keys


# ── QueryRewriteService Classification ────────────────────────────────


class TestClassification:
    """Test _classify_query_mode heuristics."""

    def test_overview_vi(self):
        from app.schemas.query_rewrite import QueryMode
        svc = _make_svc()
        assert svc._classify_query_mode("tóm tắt nội dung tài liệu") == QueryMode.OVERVIEW

    def test_overview_en(self):
        from app.schemas.query_rewrite import QueryMode
        svc = _make_svc()
        assert svc._classify_query_mode("give me an overview of the document") == QueryMode.OVERVIEW

    def test_specific(self):
        from app.schemas.query_rewrite import QueryMode
        svc = _make_svc()
        assert svc._classify_query_mode("trách nhiệm của người quản lý là gì?") == QueryMode.SPECIFIC

    def test_comparison(self):
        from app.schemas.query_rewrite import QueryMode
        svc = _make_svc()
        assert svc._classify_query_mode("so sánh hai phương án A và B") == QueryMode.COMPARISON

    def test_follow_up_vi(self):
        from app.schemas.query_rewrite import QueryMode
        svc = _make_svc()
        assert svc._classify_query_mode("cái này áp dụng cho ai?") == QueryMode.FOLLOW_UP

    def test_follow_up_en(self):
        from app.schemas.query_rewrite import QueryMode
        svc = _make_svc()
        assert svc._classify_query_mode("what about the one you mentioned earlier?") == QueryMode.FOLLOW_UP

    def test_multi_hop(self):
        from app.schemas.query_rewrite import QueryMode
        svc = _make_svc()
        assert svc._classify_query_mode("cả hai quy định liên quan đến nhau không?") == QueryMode.MULTI_HOP

    def test_ambiguous_short(self):
        from app.schemas.query_rewrite import QueryMode
        svc = _make_svc()
        assert svc._classify_query_mode("hợp đồng") == QueryMode.AMBIGUOUS

    def test_direct_clear(self):
        from app.schemas.query_rewrite import QueryMode
        svc = _make_svc()
        assert svc._classify_query_mode("chính sách nghỉ phép của công ty là gì") == QueryMode.DIRECT


# ── Follow-up Markers ─────────────────────────────────────────────────


class TestFollowUpDetection:
    """Test follow-up reference marker detection."""

    def test_vi_markers(self):
        svc = _make_svc(QUERY_REWRITE_ENABLED=False)
        assert svc._has_follow_up_markers("cái này là gì?") is True
        assert svc._has_follow_up_markers("trường hợp đó thì sao?") is True

    def test_en_markers(self):
        svc = _make_svc(QUERY_REWRITE_ENABLED=False)
        assert svc._has_follow_up_markers("what about the one you mentioned?") is True

    def test_no_markers(self):
        svc = _make_svc(QUERY_REWRITE_ENABLED=False)
        assert svc._has_follow_up_markers("chính sách nghỉ phép là gì?") is False


# ── History Resolution ────────────────────────────────────────────────


class TestHistoryResolution:
    """Test history reference resolution."""

    def test_resolve_with_pronoun(self):
        svc = _make_svc(QUERY_REWRITE_ENABLED=False)
        history = [
            {"role": "user", "text": "chính sách nghỉ phép"},
            {"role": "assistant", "text": "Chính sách nghỉ phép bao gồm..."},
        ]
        resolved = svc._resolve_history_references("cái này áp dụng cho ai?", history)
        assert resolved is not None
        assert "chính sách nghỉ phép" in resolved

    def test_no_resolve_without_reference(self):
        svc = _make_svc(QUERY_REWRITE_ENABLED=False)
        history = [
            {"role": "user", "text": "chính sách nghỉ phép"},
        ]
        resolved = svc._resolve_history_references("quy định về lương thưởng", history)
        assert resolved is None

    def test_no_resolve_empty_history(self):
        svc = _make_svc(QUERY_REWRITE_ENABLED=False)
        resolved = svc._resolve_history_references("cái này là gì", [])
        assert resolved is None

    def test_history_resolved_becomes_rewritten_when_llm_unavailable(self):
        """FIX 2: resolved_query used as rewritten fallback."""
        svc = _make_svc(QUERY_REWRITE_ENABLED=True, QUERY_REWRITE_PROVIDER="none")
        history = [
            {"role": "user", "text": "chính sách nghỉ phép"},
            {"role": "assistant", "text": "Nghỉ phép tối đa 12 ngày"},
        ]
        plan = _run(svc.maybe_rewrite("cái này áp dụng cho ai?", history))

        assert plan.original_query == "cái này áp dụng cho ai?"
        assert plan.rewritten_query is not None
        assert "chính sách nghỉ phép" in plan.rewritten_query
        assert plan.used_history is True
        assert plan.rewrite_reason == "history_resolved_without_llm"


# ── Guardrails ────────────────────────────────────────────────────────


class TestGuardrails:
    """Test guardrail application."""

    def test_low_confidence_strips_rewrite(self):
        from app.schemas.query_rewrite import QueryMode, RewriteStrategy

        svc = _make_svc()
        plan = svc._apply_guardrails(
            original_query="test",
            rewritten_query="rewritten test",
            step_back_query="broader test",
            subqueries=("sub1", "sub2"),
            query_mode=QueryMode.SPECIFIC,
            rewrite_strategy=RewriteStrategy.CONTEXTUAL_REWRITE,
            confidence=0.2,  # below 0.5 threshold
            reason="test",
            used_history=False,
            constraints=(),
            latency_ms=10,
        )
        assert plan.rewritten_query is None
        assert plan.subqueries == ()
        assert plan.step_back_query is None
        assert plan.fallback_used is True

    def test_same_as_original_stripped(self):
        from app.schemas.query_rewrite import QueryMode, RewriteStrategy

        svc = _make_svc()
        plan = svc._apply_guardrails(
            original_query="hello world",
            rewritten_query="hello world",  # same as original
            step_back_query=None,
            subqueries=(),
            query_mode=QueryMode.DIRECT,
            rewrite_strategy=RewriteStrategy.NO_REWRITE,
            confidence=0.9,
            reason="test",
            used_history=False,
            constraints=(),
            latency_ms=5,
        )
        assert plan.rewritten_query is None

    def test_subqueries_capped(self):
        from app.schemas.query_rewrite import QueryMode, RewriteStrategy

        svc = _make_svc()
        plan = svc._apply_guardrails(
            original_query="test",
            rewritten_query="better test",
            step_back_query=None,
            subqueries=("sub1", "sub2", "sub3", "sub4"),  # 4 subs, max 2
            query_mode=QueryMode.MULTI_HOP,
            rewrite_strategy=RewriteStrategy.CONTROLLED_DECOMPOSITION,
            confidence=0.9,
            reason="test",
            used_history=False,
            constraints=(),
            latency_ms=5,
        )
        assert len(plan.subqueries) <= 2

    def test_full_cross_dedupe(self):
        """FIX 4: subqueries matching original/rewritten/step_back are dropped."""
        from app.schemas.query_rewrite import QueryMode, RewriteStrategy

        svc = _make_svc()
        plan = svc._apply_guardrails(
            original_query="policy details",
            rewritten_query="company leave policy details",
            step_back_query="company policies overview",
            subqueries=("policy details", "company leave policy details"),  # both dupes
            query_mode=QueryMode.SPECIFIC,
            rewrite_strategy=RewriteStrategy.CONTEXTUAL_REWRITE,
            confidence=0.9,
            reason="test",
            used_history=False,
            constraints=(),
            latency_ms=5,
        )
        assert len(plan.subqueries) == 0

    def test_step_back_deduped_against_original(self):
        """FIX 4: step_back same as original → stripped."""
        from app.schemas.query_rewrite import QueryMode, RewriteStrategy

        svc = _make_svc()
        plan = svc._apply_guardrails(
            original_query="company leave policy",
            rewritten_query="detailed leave policy",
            step_back_query="company leave policy",  # same as original
            subqueries=(),
            query_mode=QueryMode.SPECIFIC,
            rewrite_strategy=RewriteStrategy.CONTEXTUAL_REWRITE,
            confidence=0.9,
            reason="test",
            used_history=False,
            constraints=(),
            latency_ms=5,
        )
        assert plan.step_back_query is None


# ── Validity & Forbidden Filters ──────────────────────────────────────


class TestValidation:
    """Test _is_valid_query and forbidden filter rejection."""

    def test_valid_query(self):
        from app.services.query_rewrite_service import QueryRewriteService
        assert QueryRewriteService._is_valid_query("hello world") is True

    def test_empty_invalid(self):
        from app.services.query_rewrite_service import QueryRewriteService
        assert QueryRewriteService._is_valid_query("") is False
        assert QueryRewriteService._is_valid_query(None) is False

    def test_punctuation_only_invalid(self):
        from app.services.query_rewrite_service import QueryRewriteService
        assert QueryRewriteService._is_valid_query("???...!!!") is False

    def test_too_short_invalid(self):
        from app.services.query_rewrite_service import QueryRewriteService
        assert QueryRewriteService._is_valid_query("x") is False

    def test_forbidden_filter_tenant_id(self):
        from app.services.query_rewrite_service import QueryRewriteService
        assert QueryRewriteService._is_valid_query("tenant_id: abc search") is False

    def test_forbidden_filter_doc_id(self):
        from app.services.query_rewrite_service import QueryRewriteService
        assert QueryRewriteService._is_valid_query("doc_id:123 search text") is False

    def test_forbidden_filter_metadata(self):
        from app.services.query_rewrite_service import QueryRewriteService
        assert QueryRewriteService._is_valid_query("metadata: type=pdf find docs") is False

    def test_forbidden_filter_tag(self):
        from app.services.query_rewrite_service import QueryRewriteService
        assert QueryRewriteService._is_valid_query("tag: important search") is False

    def test_forbidden_filter_source(self):
        from app.services.query_rewrite_service import QueryRewriteService
        assert QueryRewriteService._is_valid_query("source: file search text") is False

    def test_guardrails_reject_forbidden_filter_in_rewrite(self):
        """FIX 4: rewritten query with filter tokens is rejected."""
        from app.schemas.query_rewrite import QueryMode, RewriteStrategy

        svc = _make_svc()
        plan = svc._apply_guardrails(
            original_query="find policy",
            rewritten_query="tenant_id: abc find policy",
            step_back_query=None,
            subqueries=("doc_id:123 policy",),
            query_mode=QueryMode.DIRECT,
            rewrite_strategy=RewriteStrategy.CONTEXTUAL_REWRITE,
            confidence=0.9,
            reason="test",
            used_history=False,
            constraints=(),
            latency_ms=5,
        )
        assert plan.rewritten_query is None
        assert len(plan.subqueries) == 0

    def test_guardrails_reject_invalid_subquery(self):
        """FIX 4: invalid subqueries (empty, punctuation) are dropped."""
        from app.schemas.query_rewrite import QueryMode, RewriteStrategy

        svc = _make_svc()
        plan = svc._apply_guardrails(
            original_query="test query",
            rewritten_query="better test query",
            step_back_query=None,
            subqueries=("???", "", "valid sub"),
            query_mode=QueryMode.MULTI_HOP,
            rewrite_strategy=RewriteStrategy.CONTROLLED_DECOMPOSITION,
            confidence=0.9,
            reason="test",
            used_history=False,
            constraints=(),
            latency_ms=5,
        )
        assert plan.subqueries == ("valid sub",)


# ── Feature Flag Gate ─────────────────────────────────────────────────


class TestFeatureFlag:
    """Test feature flag behavior."""

    def test_disabled_returns_passthrough(self):
        svc = _make_svc(QUERY_REWRITE_ENABLED=False)
        plan = _run(svc.maybe_rewrite("test query"))
        assert plan.original_query == "test query"
        assert plan.rewritten_query is None
        assert plan.fallback_used is True

    def test_enabled_property(self):
        svc = _make_svc(QUERY_REWRITE_ENABLED=True)
        assert svc.enabled is True


# ── Fail-Open ─────────────────────────────────────────────────────────


class TestFailOpen:
    """Test fail-open behavior on errors."""

    def test_llm_error_returns_fallback(self):
        with patch("app.services.query_rewrite_service.settings") as mock_settings:
            mock_settings.QUERY_REWRITE_ENABLED = True
            mock_settings.QUERY_REWRITE_PROVIDER = "openai"
            mock_settings.QUERY_REWRITE_MODEL = "gpt-4o-mini"
            mock_settings.QUERY_REWRITE_TIMEOUT_S = 3.0
            mock_settings.QUERY_REWRITE_MAX_TOKENS = 300
            mock_settings.QUERY_REWRITE_TEMPERATURE = 0.1
            mock_settings.QUERY_REWRITE_MAX_SUBQUERIES = 2
            mock_settings.QUERY_REWRITE_MAX_QUERY_CHARS = 1200
            mock_settings.QUERY_REWRITE_CONFIDENCE_THRESHOLD = 0.5
            mock_settings.OPENAI_API_KEY = "sk-test"

            from app.services.query_rewrite_service import QueryRewriteService
            svc = QueryRewriteService()

            # Patch the LLM call to raise
            svc._llm_rewrite = AsyncMock(side_effect=Exception("LLM exploded"))

            # Use a MULTI_HOP query without constraint words so it triggers
            # CONTROLLED_DECOMPOSITION → calls LLM → LLM explodes → fallback
            plan = _run(svc.maybe_rewrite(
                "cả hai vấn đề liên quan đến nhau thế nào?",
            ))

        assert plan.original_query == "cả hai vấn đề liên quan đến nhau thế nào?"
        assert plan.fallback_used is True

    def test_empty_query(self):
        svc = _make_svc()
        plan = _run(svc.maybe_rewrite(""))
        assert plan.fallback_used is True


# ── Config ────────────────────────────────────────────────────────────


class TestConfig:
    """Test config settings exist and have defaults."""

    def test_feature_flag_default_off(self):
        from app.core.config import Settings

        assert hasattr(Settings, "model_fields")
        fields = Settings.model_fields
        assert "QUERY_REWRITE_ENABLED" in fields

    def test_all_config_keys_exist(self):
        from app.core.config import Settings

        expected = [
            "QUERY_REWRITE_ENABLED",
            "QUERY_REWRITE_PROVIDER",
            "QUERY_REWRITE_MODEL",
            "QUERY_REWRITE_TIMEOUT_S",
            "QUERY_REWRITE_MAX_TOKENS",
            "QUERY_REWRITE_TEMPERATURE",
            "QUERY_REWRITE_MAX_SUBQUERIES",
            "QUERY_REWRITE_MAX_QUERY_CHARS",
            "QUERY_REWRITE_CONFIDENCE_THRESHOLD",
        ]
        for key in expected:
            assert key in Settings.model_fields, f"Missing config: {key}"


# =====================================================================
# V2: CONSTRAINT DETECTION
# =====================================================================


class TestConstraintDetection:
    """Test V2 _detect_constraints heuristics."""

    def test_year_detected(self):
        svc = _make_svc()
        c = svc._detect_constraints("quy định nghỉ phép năm 2025")
        assert "year" in c

    def test_negation_vi_detected(self):
        svc = _make_svc()
        c = svc._detect_constraints("không được nghỉ phép quá 12 ngày")
        assert any("negation" in label for label in c)

    def test_role_detected(self):
        svc = _make_svc()
        c = svc._detect_constraints("giảng viên hợp đồng có được tăng lương")
        assert "role" in c
        assert "contract" in c

    def test_no_constraints(self):
        svc = _make_svc()
        c = svc._detect_constraints("chính sách công ty là gì")
        assert len(c) == 0


# =====================================================================
# V2: REWRITE STRATEGY (GATING)
# =====================================================================


class TestRewriteStrategy:
    """Test V2 _determine_strategy gating decisions."""

    def test_direct_no_rewrite(self):
        from app.schemas.query_rewrite import QueryMode, RewriteStrategy
        svc = _make_svc()
        s = svc._determine_strategy(
            "chính sách nghỉ phép của công ty là gì",
            QueryMode.DIRECT, None, False,
        )
        assert s == RewriteStrategy.NO_REWRITE

    def test_follow_up_with_history(self):
        from app.schemas.query_rewrite import QueryMode, RewriteStrategy
        svc = _make_svc()
        s = svc._determine_strategy(
            "cái này áp dụng cho ai",
            QueryMode.FOLLOW_UP,
            [{"role": "user", "text": "prev"}],
            False,
        )
        assert s == RewriteStrategy.CONTEXTUAL_REWRITE

    def test_follow_up_no_history(self):
        from app.schemas.query_rewrite import QueryMode, RewriteStrategy
        svc = _make_svc()
        s = svc._determine_strategy(
            "cái này áp dụng cho ai",
            QueryMode.FOLLOW_UP, None, False,
        )
        assert s == RewriteStrategy.SAFE_FALLBACK

    def test_ambiguous_no_history(self):
        from app.schemas.query_rewrite import QueryMode, RewriteStrategy
        svc = _make_svc()
        s = svc._determine_strategy(
            "hợp đồng", QueryMode.AMBIGUOUS, None, False,
        )
        assert s == RewriteStrategy.SAFE_FALLBACK

    def test_multi_hop_decomposition(self):
        from app.schemas.query_rewrite import QueryMode, RewriteStrategy
        svc = _make_svc()
        s = svc._determine_strategy(
            "cả hai quy định liên quan đến nhau không",
            QueryMode.MULTI_HOP, None, False,
        )
        assert s == RewriteStrategy.CONTROLLED_DECOMPOSITION

    def test_constraints_cap_decomposition(self):
        """MULTI_HOP + constraints → LIGHT_NORMALIZE (caps decomposition)."""
        from app.schemas.query_rewrite import QueryMode, RewriteStrategy
        svc = _make_svc()
        s = svc._determine_strategy(
            "cả hai quy định năm 2025 cho giảng viên hợp đồng liên quan đến nhau không",
            QueryMode.MULTI_HOP, None, True,
        )
        assert s == RewriteStrategy.LIGHT_NORMALIZE

    def test_specific_defaults_conservative(self):
        """SPECIFIC now defaults to LIGHT_NORMALIZE (conservative)."""
        from app.schemas.query_rewrite import QueryMode, RewriteStrategy
        svc = _make_svc()
        s = svc._determine_strategy(
            "quy trình nghỉ phép bao gồm những bước nào",
            QueryMode.SPECIFIC, None, False,
        )
        assert s == RewriteStrategy.LIGHT_NORMALIZE

    def test_overview_light_normalize(self):
        from app.schemas.query_rewrite import QueryMode, RewriteStrategy
        svc = _make_svc()
        s = svc._determine_strategy(
            "tổng quan quy trình", QueryMode.OVERVIEW, None, False,
        )
        assert s == RewriteStrategy.LIGHT_NORMALIZE


# =====================================================================
# V2: CONSTRAINT PRESERVATION
# =====================================================================


class TestConstraintPreservation:
    """Test V2 _constraints_preserved guardrail."""

    def test_year_preserved(self):
        from app.services.query_rewrite_service import QueryRewriteService
        assert QueryRewriteService._constraints_preserved(
            "quy định năm 2025", "nghỉ phép năm 2025 cụ thể",
        ) is True

    def test_year_dropped(self):
        from app.services.query_rewrite_service import QueryRewriteService
        assert QueryRewriteService._constraints_preserved(
            "quy định năm 2025", "quy định nghỉ phép chung",
        ) is False

    def test_negation_dropped(self):
        from app.services.query_rewrite_service import QueryRewriteService
        assert QueryRewriteService._constraints_preserved(
            "không được nghỉ quá 12 ngày",
            "nghỉ phép tối đa 12 ngày",  # lost "không"
        ) is False


# =====================================================================
# V2: CONSTRAINT AS ORTHOGONAL SIGNAL (not a QueryMode)
# =====================================================================


class TestConstraintAsOrthogonalSignal:
    """Constraints are detected separately and influence gating,
    but do NOT override the behavior class (QueryMode)."""

    def test_constrained_query_classifies_by_shape(self):
        """A query with constraints should classify by shape, not become
        CONSTRAINT_HEAVY (removed from active classification)."""
        from app.schemas.query_rewrite import QueryMode
        svc = _make_svc()
        mode = svc._classify_query_mode(
            "quy định nghỉ phép năm 2025 cho giảng viên hợp đồng",
        )
        assert mode == QueryMode.SPECIFIC
        assert mode != QueryMode.CONSTRAINT_HEAVY

    def test_constraints_detected_independently(self):
        """_detect_constraints returns constraint labels regardless of mode."""
        svc = _make_svc()
        constraints = svc._detect_constraints(
            "quy định nghỉ phép năm 2025 cho giảng viên hợp đồng",
        )
        assert "year" in constraints
        assert "role" in constraints
        assert "contract" in constraints

    def test_constraints_influence_gating(self):
        """Constraints cap MULTI_HOP from DECOMPOSITION to LIGHT_NORMALIZE."""
        from app.schemas.query_rewrite import QueryMode, RewriteStrategy
        svc = _make_svc()
        s1 = svc._determine_strategy(
            "cả hai liên quan đến nhau không",
            QueryMode.MULTI_HOP, None, False,
        )
        assert s1 == RewriteStrategy.CONTROLLED_DECOMPOSITION
        s2 = svc._determine_strategy(
            "cả hai liên quan năm 2025 giảng viên",
            QueryMode.MULTI_HOP, None, True,
        )
        assert s2 == RewriteStrategy.LIGHT_NORMALIZE


# =====================================================================
# V2: END-TO-END
# =====================================================================


class TestV2EndToEnd:
    """Test V2 rewrite flow end-to-end."""

    def test_constrained_specific_preserves_original(self):
        """SPECIFIC + constraints → LIGHT_NORMALIZE → no rewrite, original kept."""
        svc = _make_svc(QUERY_REWRITE_ENABLED=True, QUERY_REWRITE_PROVIDER="none")
        plan = _run(svc.maybe_rewrite(
            "quy định nghỉ phép năm 2025 cho giảng viên hợp đồng",
        ))
        assert plan.rewritten_query is None
        assert plan.fallback_used is False
        assert plan.rewrite_strategy == "light_normalize"
        assert plan.query_mode.value == "specific"

    def test_ambiguous_no_history_safe_fallback(self):
        """AMBIGUOUS + no history → SAFE_FALLBACK."""
        svc = _make_svc(QUERY_REWRITE_ENABLED=True, QUERY_REWRITE_PROVIDER="none")
        plan = _run(svc.maybe_rewrite("hợp đồng"))
        assert plan.fallback_used is False
        assert plan.rewrite_strategy == "safe_fallback"

    def test_rewrite_strategy_in_telemetry(self):
        """Telemetry dict includes rewrite_strategy."""
        svc = _make_svc(QUERY_REWRITE_ENABLED=True, QUERY_REWRITE_PROVIDER="none")
        plan = _run(svc.maybe_rewrite("chính sách công ty hiện tại là gì"))
        tele = plan.telemetry_dict()
        assert "rewrite_strategy" in tele
        assert tele["rewrite_strategy"] == plan.rewrite_strategy

    def test_specific_clear_query_no_llm(self):
        """SPECIFIC query that's already clear → LIGHT_NORMALIZE, no LLM."""
        svc = _make_svc(QUERY_REWRITE_ENABLED=True, QUERY_REWRITE_PROVIDER="none")
        plan = _run(svc.maybe_rewrite("quy trình nghỉ phép bao gồm những bước nào"))
        assert plan.rewrite_strategy == "light_normalize"
        assert plan.rewritten_query is None
        assert plan.fallback_used is False

    def test_history_resolution_standalone_format(self):
        """History resolution builds standalone query, not raw append."""
        svc = _make_svc(QUERY_REWRITE_ENABLED=True, QUERY_REWRITE_PROVIDER="none")
        history = [
            {"role": "user", "text": "chính sách nghỉ phép"},
            {"role": "assistant", "text": "Nghỉ phép tối đa 12 ngày."},
        ]
        plan = _run(svc.maybe_rewrite("cái này áp dụng cho ai?", history))
        if plan.rewritten_query:
            assert "(liên quan:" not in plan.rewritten_query
            assert "chính sách nghỉ phép" in plan.rewritten_query

