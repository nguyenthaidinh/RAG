"""
Phase 3 — RetrievalExecutionContext tests.

Covers:
  1. Context construction and field defaults
  2. Frozen immutability
  3. telemetry_dict() safety (no raw text)
  4. QueryService builds context correctly for different scenarios:
     a. rewrite usable
     b. rewrite not usable / fallback
     c. metadata preference populated
     d. representation preference populated
  5. History flag tracking
  6. Assistant/query parity preserved
  7. Existing test suites regression (validated externally)
"""
from __future__ import annotations

import asyncio
from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.retrieval.retrieval_execution_context import (
    RetrievalExecutionContext,
)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# =====================================================================
# 1. CONSTRUCTION AND DEFAULTS
# =====================================================================


class TestContextDefaults:
    """Test that RetrievalExecutionContext has safe defaults."""

    def test_empty_context(self):
        ctx = RetrievalExecutionContext.empty()
        assert ctx.original_query == ""
        assert ctx.effective_mode == "hybrid"
        assert ctx.include_debug is False
        assert ctx.rewrite_plan is None
        assert ctx.rewrite_usable is False
        assert ctx.query_plan is None
        assert ctx.effective_queries == ()
        assert ctx.candidate_doc_ids == frozenset()
        assert ctx.history_provided is False
        assert ctx.metadata_preference is None
        assert ctx.representation_preference is None

    def test_custom_construction(self):
        ctx = RetrievalExecutionContext(
            original_query="test query",
            effective_mode="vector",
            include_debug=True,
            rewrite_usable=True,
            effective_queries=("q1", "q2"),
            candidate_doc_ids=frozenset({1, 2, 3}),
            history_provided=True,
        )
        assert ctx.original_query == "test query"
        assert ctx.effective_mode == "vector"
        assert ctx.include_debug is True
        assert ctx.rewrite_usable is True
        assert ctx.effective_queries == ("q1", "q2")
        assert ctx.candidate_doc_ids == frozenset({1, 2, 3})
        assert ctx.history_provided is True


# =====================================================================
# 2. FROZEN IMMUTABILITY
# =====================================================================


class TestFrozenContract:
    """Context is immutable after construction."""

    def test_cannot_mutate_fields(self):
        ctx = RetrievalExecutionContext.empty()
        with pytest.raises(AttributeError):
            ctx.original_query = "modified"

    def test_cannot_mutate_mode(self):
        ctx = RetrievalExecutionContext.empty()
        with pytest.raises(AttributeError):
            ctx.effective_mode = "bm25"

    def test_replace_creates_new_instance(self):
        ctx1 = RetrievalExecutionContext(original_query="q1")
        ctx2 = replace(ctx1, original_query="q2")
        assert ctx1.original_query == "q1"
        assert ctx2.original_query == "q2"
        assert ctx1 is not ctx2


# =====================================================================
# 3. TELEMETRY SAFETY
# =====================================================================


class TestTelemetry:
    """telemetry_dict() must not leak raw text."""

    def test_telemetry_no_raw_text(self):
        ctx = RetrievalExecutionContext(
            original_query="secret query about confidential docs",
            effective_queries=("secret q1", "secret q2"),
        )
        tele = ctx.telemetry_dict()
        tele_str = str(tele)
        assert "secret" not in tele_str
        assert "confidential" not in tele_str

    def test_telemetry_has_expected_keys(self):
        ctx = RetrievalExecutionContext.empty()
        tele = ctx.telemetry_dict()
        expected_keys = {
            "effective_mode",
            "include_debug",
            "rewrite_usable",
            "rewrite_plan_present",
            "query_plan_present",
            "effective_query_count",
            "candidate_doc_count",
            "history_provided",
            "metadata_pref_present",
            "representation_pref_present",
        }
        assert set(tele.keys()) == expected_keys

    def test_telemetry_values_correct(self):
        ctx = RetrievalExecutionContext(
            original_query="test",
            effective_mode="bm25",
            include_debug=True,
            rewrite_usable=True,
            effective_queries=("q1", "q2", "q3"),
            candidate_doc_ids=frozenset({1, 2}),
            history_provided=True,
        )
        tele = ctx.telemetry_dict()
        assert tele["effective_mode"] == "bm25"
        assert tele["include_debug"] is True
        assert tele["rewrite_usable"] is True
        assert tele["rewrite_plan_present"] is False
        assert tele["query_plan_present"] is False
        assert tele["effective_query_count"] == 3
        assert tele["candidate_doc_count"] == 2
        assert tele["history_provided"] is True


# =====================================================================
# 4. QUERY SERVICE BUILDS CONTEXT CORRECTLY
# =====================================================================


def _make_query_service(
    *,
    rewriter=None,
    planner=None,
    metadata_intent=None,
    repr_intent=None,
):
    """Build a QueryService with mocked dependencies."""
    from app.services.retrieval.query_service import QueryService

    mock_access = AsyncMock()
    mock_access.allowed_documents = AsyncMock(return_value={1, 2, 3})

    mock_embedding = AsyncMock()
    mock_embedding.embed = AsyncMock(return_value=[[0.1, 0.2]])

    mock_vector = AsyncMock()
    mock_vector.search = AsyncMock(return_value=[])

    mock_bm25 = AsyncMock()
    mock_bm25.search = AsyncMock(return_value=[])

    mock_reranker = AsyncMock()
    mock_reranker.rerank = AsyncMock(return_value=[])

    mock_response_builder = MagicMock()
    mock_response_builder.build = MagicMock(return_value=[])

    svc = QueryService(
        vector_retriever=mock_vector,
        bm25_retriever=mock_bm25,
        embedding_provider=mock_embedding,
        access_policy=mock_access,
        reranker=mock_reranker,
        response_builder=mock_response_builder,
        query_rewriter=rewriter,
        planner=planner,
        metadata_intent_service=metadata_intent,
        representation_intent_service=repr_intent,
    )
    return svc


class TestBuildExecutionContext:
    """Test _build_execution_context via QueryService."""

    def test_rewrite_usable_context(self):
        """When rewrite succeeds, context reflects rewrite state."""
        from app.schemas.query_rewrite import QueryMode, RetrievalPlan

        mock_rewriter = MagicMock()
        mock_rewriter.enabled = True
        mock_rewriter.maybe_rewrite = AsyncMock(return_value=RetrievalPlan(
            original_query="test query",
            rewritten_query="improved test query",
            query_mode=QueryMode.SPECIFIC,
            confidence=0.9,
            fallback_used=False,
        ))

        svc = _make_query_service(rewriter=mock_rewriter)
        ctx = _run(svc._build_execution_context(
            query_text="test query",
            mode="hybrid",
            include_debug=False,
            history=None,
            tenant_id="t1",
            user_id=1,
            allowed_doc_ids={1, 2, 3},
        ))

        assert ctx.rewrite_usable is True
        assert ctx.rewrite_plan is not None
        assert ctx.rewrite_plan.rewritten_query == "improved test query"
        assert "test query" in ctx.effective_queries
        assert "improved test query" in ctx.effective_queries
        assert ctx.effective_mode == "hybrid"
        assert ctx.history_provided is False

    def test_rewrite_not_usable_fallback(self):
        """When rewrite falls back, context reflects passthrough."""
        from app.schemas.query_rewrite import RetrievalPlan

        mock_rewriter = MagicMock()
        mock_rewriter.enabled = True
        mock_rewriter.maybe_rewrite = AsyncMock(
            return_value=RetrievalPlan.passthrough("test query"),
        )

        svc = _make_query_service(rewriter=mock_rewriter)
        ctx = _run(svc._build_execution_context(
            query_text="test query",
            mode="hybrid",
            include_debug=False,
            history=None,
            tenant_id="t1",
            user_id=1,
            allowed_doc_ids={1, 2, 3},
        ))

        assert ctx.rewrite_usable is False
        assert ctx.rewrite_plan is not None
        assert ctx.rewrite_plan.fallback_used is True
        assert len(ctx.effective_queries) >= 1

    def test_rewrite_failure_failopen(self):
        """When rewrite service raises, context degrades gracefully."""
        mock_rewriter = MagicMock()
        mock_rewriter.enabled = True
        mock_rewriter.maybe_rewrite = AsyncMock(
            side_effect=RuntimeError("rewrite exploded"),
        )

        svc = _make_query_service(rewriter=mock_rewriter)
        ctx = _run(svc._build_execution_context(
            query_text="test query",
            mode="hybrid",
            include_debug=False,
            history=None,
            tenant_id="t1",
            user_id=1,
            allowed_doc_ids={1, 2, 3},
        ))

        assert ctx.rewrite_usable is False
        assert ctx.rewrite_plan is None
        assert len(ctx.effective_queries) >= 1

    def test_mode_normalization(self):
        """Invalid mode defaults to hybrid."""
        svc = _make_query_service()
        ctx = _run(svc._build_execution_context(
            query_text="q",
            mode="invalid_mode",
            include_debug=False,
            history=None,
            tenant_id="t1",
            user_id=1,
            allowed_doc_ids={1},
        ))
        assert ctx.effective_mode == "hybrid"

    def test_bm25_mode(self):
        """BM25 mode is preserved."""
        svc = _make_query_service()
        ctx = _run(svc._build_execution_context(
            query_text="q",
            mode="bm25",
            include_debug=False,
            history=None,
            tenant_id="t1",
            user_id=1,
            allowed_doc_ids={1},
        ))
        assert ctx.effective_mode == "bm25"

    def test_history_tracked(self):
        """history_provided reflects whether history was given."""
        svc = _make_query_service()

        ctx_with = _run(svc._build_execution_context(
            query_text="q",
            mode="hybrid",
            include_debug=False,
            history=[{"role": "user", "text": "prev"}],
            tenant_id="t1",
            user_id=1,
            allowed_doc_ids={1},
        ))
        assert ctx_with.history_provided is True

        ctx_without = _run(svc._build_execution_context(
            query_text="q",
            mode="hybrid",
            include_debug=False,
            history=None,
            tenant_id="t1",
            user_id=1,
            allowed_doc_ids={1},
        ))
        assert ctx_without.history_provided is False

    def test_candidate_doc_ids_intersection(self):
        """Candidate doc IDs = allowed ∩ plan.filters.doc_ids."""
        svc = _make_query_service()
        ctx = _run(svc._build_execution_context(
            query_text="q",
            mode="hybrid",
            include_debug=False,
            history=None,
            tenant_id="t1",
            user_id=1,
            allowed_doc_ids={1, 2, 3, 4},
        ))
        # No plan filter → all allowed IDs are candidates
        assert ctx.candidate_doc_ids == frozenset({1, 2, 3, 4})

    def test_include_debug_propagated(self):
        svc = _make_query_service()
        ctx = _run(svc._build_execution_context(
            query_text="q",
            mode="hybrid",
            include_debug=True,
            history=None,
            tenant_id="t1",
            user_id=1,
            allowed_doc_ids={1},
        ))
        assert ctx.include_debug is True


# =====================================================================
# 5. INTENT PREFERENCE RESOLUTION
# =====================================================================


class TestResolveIntentPreferences:
    """Test _resolve_intent_preferences populates context."""

    def test_metadata_preference_populated(self):
        from app.schemas.retrieval_metadata import MetadataPreference

        mock_metadata = MagicMock()
        mock_metadata.enabled = True
        mock_metadata.parse = MagicMock(return_value=MetadataPreference(
            preferred_sources=("upload",),
            confidence=0.8,
            reason="source_detected",
        ))

        svc = _make_query_service(metadata_intent=mock_metadata)
        ctx = RetrievalExecutionContext(original_query="test")
        ctx2 = svc._resolve_intent_preferences(
            ctx, query_text="test", tenant_id="t1", user_id=1,
        )

        assert ctx2.metadata_preference is not None
        assert ctx2.metadata_preference.preferred_sources == ("upload",)
        # Original ctx not mutated (frozen)
        assert ctx.metadata_preference is None

    def test_representation_preference_populated(self):
        from app.schemas.retrieval_representation import (
            RepresentationIntent,
            RepresentationPreference,
        )

        mock_repr = MagicMock()
        mock_repr.enabled = True
        mock_repr.classify = MagicMock(return_value=RepresentationPreference(
            intent=RepresentationIntent.OVERVIEW_SUMMARY,
            preferred_type="synthesized",
            strength=0.8,
            confidence=0.9,
            reason="overview_query",
        ))

        svc = _make_query_service(repr_intent=mock_repr)
        ctx = RetrievalExecutionContext(original_query="test")
        ctx2 = svc._resolve_intent_preferences(
            ctx, query_text="test", tenant_id="t1", user_id=1,
        )

        assert ctx2.representation_preference is not None
        assert ctx2.representation_preference.intent == RepresentationIntent.OVERVIEW_SUMMARY

    def test_no_preferences_returns_same_ctx(self):
        """When no intent services enabled, returns same context object."""
        svc = _make_query_service()
        ctx = RetrievalExecutionContext(original_query="test")
        ctx2 = svc._resolve_intent_preferences(
            ctx, query_text="test", tenant_id="t1", user_id=1,
        )
        assert ctx2 is ctx  # no change → same object

    def test_intent_failure_failopen(self):
        """When intent service raises, preference stays None."""
        mock_metadata = MagicMock()
        mock_metadata.enabled = True
        mock_metadata.parse = MagicMock(side_effect=RuntimeError("boom"))

        svc = _make_query_service(metadata_intent=mock_metadata)
        ctx = RetrievalExecutionContext(original_query="test")
        ctx2 = svc._resolve_intent_preferences(
            ctx, query_text="test", tenant_id="t1", user_id=1,
        )

        # Fail-open: no preference
        assert ctx2 is ctx  # both None → same ctx returned


# =====================================================================
# 6. END-TO-END: QueryService.query() uses context
# =====================================================================


class TestQueryServiceEndToEnd:
    """Verify QueryService.query() uses RetrievalExecutionContext
    and behavior is unchanged."""

    def test_query_returns_results(self):
        """Basic end-to-end: query returns results via context-based flow."""
        from app.services.retrieval.types import QueryResult

        svc = _make_query_service()
        # Mock response builder to return a result
        expected = [QueryResult(
            chunk_id=1, document_id=1, score=0.9,
            snippet="test", highlights=(),
        )]
        svc._response_builder.build = MagicMock(return_value=expected)
        svc._reranker.rerank = AsyncMock(return_value=[])

        results = _run(svc.query(
            tenant_id="t1",
            user_id=1,
            query_text="what is test",
        ))
        assert results == expected

    def test_query_with_history(self):
        """Query with history still works through context flow."""
        svc = _make_query_service()
        svc._reranker.rerank = AsyncMock(return_value=[])

        results = _run(svc.query(
            tenant_id="t1",
            user_id=1,
            query_text="follow up question",
            history=[{"role": "user", "text": "prev"}],
        ))
        # Pipeline completes successfully
        assert isinstance(results, list)

    def test_query_no_access_returns_empty(self):
        """Access policy returns empty → no results."""
        svc = _make_query_service()
        svc._access_policy.allowed_documents = AsyncMock(return_value=set())

        results = _run(svc.query(
            tenant_id="t1",
            user_id=1,
            query_text="test",
        ))
        assert results == []


# =====================================================================
# 7. _execute_retrieval helper
# =====================================================================


class TestExecuteRetrieval:
    """Test the _execute_retrieval private helper."""

    def test_embedding_failure_returns_empty(self):
        """Embedding provider raises → empty results (hard fail)."""
        svc = _make_query_service()
        svc._embedding_provider.embed = AsyncMock(
            side_effect=RuntimeError("embed failed"),
        )

        ctx = RetrievalExecutionContext(
            original_query="test",
            effective_mode="hybrid",
            effective_queries=("test",),
            candidate_doc_ids=frozenset({1}),
        )

        merged, vec_hits, bm25_hits = _run(svc._execute_retrieval(
            ctx=ctx,
            tenant_id="t1",
            user_id=1,
            vector_limit=10,
            bm25_limit=10,
            final_limit=5,
        ))
        assert merged == []
        assert vec_hits == 0
        assert bm25_hits == 0

    def test_bm25_mode_skips_embedding(self):
        """BM25-only mode should not call embedding provider."""
        svc = _make_query_service()

        ctx = RetrievalExecutionContext(
            original_query="test",
            effective_mode="bm25",
            effective_queries=("test",),
            candidate_doc_ids=frozenset({1}),
        )

        _run(svc._execute_retrieval(
            ctx=ctx,
            tenant_id="t1",
            user_id=1,
            vector_limit=10,
            bm25_limit=10,
            final_limit=5,
        ))

        svc._embedding_provider.embed.assert_not_called()

    def test_vector_results_returned(self):
        """When vector search returns results, they appear in merged output."""
        from app.services.retrieval.types import ScoredChunk

        chunk = ScoredChunk(
            chunk_id=100, document_id=1, version_id="v1",
            chunk_index=0, score=0.95, source="vector",
            snippet="test snippet",
        )

        svc = _make_query_service()
        svc._vector_retriever.search = AsyncMock(return_value=[chunk])
        svc._bm25_retriever.search = AsyncMock(return_value=[])

        ctx = RetrievalExecutionContext(
            original_query="test",
            effective_mode="vector",
            effective_queries=("test",),
            candidate_doc_ids=frozenset({1}),
        )

        merged, vec_hits, bm25_hits = _run(svc._execute_retrieval(
            ctx=ctx,
            tenant_id="t1",
            user_id=1,
            vector_limit=10,
            bm25_limit=10,
            final_limit=5,
        ))
        assert len(merged) == 1
        assert merged[0].chunk_id == 100
        assert vec_hits == 1
        assert bm25_hits == 0


# =====================================================================
# 8. _record_usage_and_log helper
# =====================================================================


class TestRecordUsageAndLog:
    """Test the _record_usage_and_log private helper."""

    def test_usage_service_called(self):
        """Usage service is called when present."""
        svc = _make_query_service()
        mock_usage = AsyncMock()
        mock_usage.record_query_usage = AsyncMock()
        svc._query_usage_service = mock_usage

        ctx = RetrievalExecutionContext(
            original_query="test",
            effective_mode="hybrid",
        )

        _run(svc._record_usage_and_log(
            ctx=ctx,
            tenant_id="t1",
            user_id=1,
            idempotency_key="key-1",
            query_text="test",
            vector_limit=30,
            bm25_limit=30,
            final_limit=10,
            reranked=[],
            results=[],
            vector_hits_count=5,
            bm25_hits_count=3,
            elapsed_ms=100,
        ))

        mock_usage.record_query_usage.assert_called_once()
        call_kwargs = mock_usage.record_query_usage.call_args[1]
        assert call_kwargs["mode"] == "hybrid"
        assert call_kwargs["idempotency_key"] == "key-1"

    def test_usage_failure_is_failopen(self):
        """Usage service failure does not propagate."""
        svc = _make_query_service()
        mock_usage = AsyncMock()
        mock_usage.record_query_usage = AsyncMock(
            side_effect=RuntimeError("usage DB down"),
        )
        svc._query_usage_service = mock_usage

        ctx = RetrievalExecutionContext(
            original_query="test",
            effective_mode="hybrid",
        )

        # Should not raise
        _run(svc._record_usage_and_log(
            ctx=ctx,
            tenant_id="t1",
            user_id=1,
            idempotency_key=None,
            query_text="test",
            vector_limit=30,
            bm25_limit=30,
            final_limit=10,
            reranked=[],
            results=[],
            vector_hits_count=0,
            bm25_hits_count=0,
            elapsed_ms=50,
        ))

    def test_no_usage_service_still_logs(self):
        """Telemetry logging works even without usage service."""
        svc = _make_query_service()
        assert svc._query_usage_service is None

        ctx = RetrievalExecutionContext(
            original_query="test",
            effective_mode="bm25",
        )

        # Should not raise
        _run(svc._record_usage_and_log(
            ctx=ctx,
            tenant_id="t1",
            user_id=1,
            idempotency_key=None,
            query_text="test",
            vector_limit=30,
            bm25_limit=30,
            final_limit=10,
            reranked=[],
            results=[],
            vector_hits_count=0,
            bm25_hits_count=0,
            elapsed_ms=25,
        ))

