"""
Query orchestrator — the single entry point for retrieval.

Pipeline phases (context-first orchestration):
  0. Metadata-first retrieval using semantic JSON metadata
  0.5 Document-level ACL filter on metadata-first candidates (Phase 6)
  1. If metadata-first is good enough -> return early
  2. Input normalization (scope, limits)
  3. Access control (fail-closed)
  4. Build execution context (rewrite, plan, queries)
  5. Multi-query retrieval (BM25 / Vector / Hybrid)
  6. Re-rank (fail-open)
  7. Intent preference resolution (metadata + representation)
  8. Metadata bias application (fail-open)
  8.5 Defense-in-depth ACL sweep on reranked chunks (Phase 6)
  9. Family consolidation (fail-open)
  10. Response build + usage recording + telemetry

STRICT RULES:
  - No vendor logic
  - Direct DB access is limited to internal metadata/family lookups
  - No raw text logging (no query_text / snippets in logs)
  - No direct DB access outside scoped helpers
  - Fail-open on non-critical subsystems
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Literal

from app.core.config import settings
from app.services.embedding_provider import EmbeddingProvider
from app.services.retrieval.access_policy import AccessPolicy
from app.services.retrieval.bm25_retriever import BM25Retriever
from app.services.retrieval.hybrid import HybridConfig, HybridStrategy
from app.services.retrieval.query_plan import QueryPlan, QueryPlanFilters
from app.services.retrieval.query_planner import QueryPlanner
from app.services.retrieval.reranker import DeterministicReRanker, ReRanker
from app.services.retrieval.response_builder import ResponseBuilder
from app.services.retrieval.types import QueryResult, QueryScope, ScoredChunk, VectorFilter
from app.services.retrieval.vector_retriever import VectorRetriever
from app.services.retrieval.document_representation_selector import (
    DocumentRepresentationSelector,
    RetrievalCandidate,
)
from app.services.retrieval.retrieval_execution_context import (
    RetrievalExecutionContext,
)
from app.services.retrieval.document_access_evaluator import (
    UserAccessContext,
    extract_access_scope_from_metadata,
    filter_candidates_by_acl,
    normalize_access_scope,
    evaluate_document_access,
    resolve_user_access_context,
)

logger = logging.getLogger(__name__)

QueryMode = Literal["hybrid", "vector", "bm25"]


def _safe_int(value, default: int) -> int:
    try:
        v = int(value)
        return v if v > 0 else default
    except Exception:
        return default


class QueryService:
    """
    Orchestrates the full retrieval pipeline.
    All dependencies are injected — vendor logic stays in adapters.
    """

    __slots__ = (
        "_access_policy",
        "_embedding_provider",
        "_vector_retriever",
        "_bm25_retriever",
        "_hybrid",
        "_reranker",
        "_response_builder",
        "_query_usage_service",
        "_planner",
        "_representation_selector",
        "_session_factory",
        "_query_rewriter",
        "_metadata_intent_service",
        "_metadata_bias_reranker",
        "_representation_intent_service",
        "_metadata_first_retrieval_service",
    )

    def __init__(
        self,
        *,
        access_policy: AccessPolicy,
        embedding_provider: EmbeddingProvider,
        vector_retriever: VectorRetriever,
        bm25_retriever: BM25Retriever,
        hybrid_config: HybridConfig | None = None,
        reranker: ReRanker | None = None,
        response_builder: ResponseBuilder | None = None,
        query_usage_service=None,
        planner: QueryPlanner | None = None,
        representation_selector: DocumentRepresentationSelector | None = None,
        session_factory=None,
        query_rewriter=None,
        metadata_intent_service=None,
        metadata_bias_reranker=None,
        representation_intent_service=None,
        metadata_first_retrieval_service=None,
    ) -> None:
        self._access_policy = access_policy
        self._embedding_provider = embedding_provider
        self._vector_retriever = vector_retriever
        self._bm25_retriever = bm25_retriever
        self._hybrid = HybridStrategy(hybrid_config)
        self._reranker: ReRanker = reranker or DeterministicReRanker()
        self._response_builder = response_builder or ResponseBuilder()
        self._query_usage_service = query_usage_service
        self._planner = planner
        self._representation_selector = representation_selector
        self._query_rewriter = query_rewriter
        self._metadata_intent_service = metadata_intent_service
        self._metadata_bias_reranker = metadata_bias_reranker
        self._representation_intent_service = representation_intent_service
        self._metadata_first_retrieval_service = metadata_first_retrieval_service

        if self._metadata_first_retrieval_service is None:
            try:
                from app.services.retrieval.metadata_first_retrieval import MetadataFirstRetrievalService
                self._metadata_first_retrieval_service = MetadataFirstRetrievalService()
            except Exception:
                logger.warning("retrieval.metadata_first_init_failed", exc_info=True)
                self._metadata_first_retrieval_service = None

        if session_factory is None:
            from app.db.session import AsyncSessionLocal
            session_factory = AsyncSessionLocal
        self._session_factory = session_factory

    @staticmethod
    def _merge_keep_best(chunks: list[ScoredChunk]) -> list[ScoredChunk]:
        by_chunk_id: dict[int, ScoredChunk] = {}
        for chunk in chunks:
            prev = by_chunk_id.get(chunk.chunk_id)
            if prev is None or chunk.score > prev.score:
                by_chunk_id[chunk.chunk_id] = chunk
        merged = list(by_chunk_id.values())
        merged.sort(key=lambda c: (-c.score, c.chunk_id))
        return merged

    @staticmethod
    def _metadata_candidates_to_query_results(candidates) -> list[QueryResult]:
        """
        Convert metadata-first candidates directly into QueryResult.
        This is the simplest early-return path for JSON-first retrieval.
        """
        results: list[QueryResult] = []

        for c in candidates:
            snippet = (c.content_text or "")[:1000]
            results.append(
                QueryResult(
                    chunk_id=0,  # metadata-first returns doc-level matches, not chunk-level
                    document_id=c.document_id,
                    title=c.title,
                    source=c.source,
                    score=c.score,
                    snippet=snippet,
                    version_id="metadata-first",
                    debug={
                        "mode": "metadata_first",
                        "reasons": c.reasons,
                        "representation_type": c.representation_type,
                        "parent_document_id": c.parent_document_id,
                    },
                )
            )
        return results

    async def query(
        self,
        *,
        tenant_id: str,
        user_id: int,
        query_text: str,
        scope: QueryScope | None = None,
        vector_limit: int | None = None,
        bm25_limit: int | None = None,
        final_limit: int | None = None,
        idempotency_key: str | None = None,
        mode: QueryMode = "hybrid",
        include_debug: bool = False,
        history: list | None = None,
    ) -> list[QueryResult]:
        start = time.monotonic()

        # ─────────────────────────────
        # Phase 1: Input normalization
        # ─────────────────────────────
        scope = scope or QueryScope()
        vector_limit = _safe_int(
            vector_limit, getattr(settings, "QUERY_VECTOR_LIMIT", 30)
        )
        bm25_limit = _safe_int(
            bm25_limit, getattr(settings, "QUERY_BM25_LIMIT", 30)
        )
        final_limit = _safe_int(
            final_limit, getattr(settings, "QUERY_FINAL_LIMIT", 10)
        )

        # ─────────────────────────────
        # Normalize mode defensively
        # ─────────────────────────────
        mode_norm = "hybrid"
        if isinstance(mode, str):
            m = mode.strip().lower()
            if m in ("hybrid", "vector", "bm25"):
                mode_norm = m
            else:
                logger.warning(
                    "retrieval.invalid_mode tenant_id=%s user_id=%d mode=%s",
                    tenant_id,
                    user_id,
                    m,
                )

        # ─────────────────────────────
        # Phase 0: Metadata-first retrieval (fail-open)
        # ─────────────────────────────

        # ── Phase 6: Resolve user access context once per request ──
        # Used by both metadata-first and defense-in-depth ACL paths.
        # Fail-safe: returns user_id-only context if system context
        # is unavailable.
        acl_user_ctx = await resolve_user_access_context(
            tenant_id=tenant_id,
            user_id=user_id,
        )

        if self._metadata_first_retrieval_service is not None:
            try:
                async with self._session_factory() as db:
                    metadata_first_candidates = (
                        await self._metadata_first_retrieval_service.retrieve(
                            db=db,
                            tenant_id=tenant_id,
                            query=query_text,
                            limit=max(final_limit, 5),
                        )
                    )

                # ── Phase 6: ACL-filter BEFORE good-enough check ─────
                # Without this, restricted docs could trigger early-return
                # and bypass the normal access policy phase entirely.
                metadata_first_candidates = self._apply_acl_to_metadata_candidates(
                    metadata_first_candidates,
                    acl_user_ctx=acl_user_ctx,
                )

                if self._metadata_first_retrieval_service.is_good_enough(
                    metadata_first_candidates
                ):
                    metadata_first_results = self._metadata_candidates_to_query_results(
                        metadata_first_candidates[:final_limit]
                    )

                    elapsed_ms = int((time.monotonic() - start) * 1000)

                    if self._query_usage_service is not None:
                        try:
                            max_len = _safe_int(
                                getattr(settings, "QUERY_USAGE_SNIPPET_MAX_LEN", 1000),
                                1000,
                            )

                            context_texts = [
                                (r.snippet[:max_len] if getattr(r, "snippet", None) else "")
                                for r in metadata_first_results
                            ]

                            await self._query_usage_service.record_query_usage(
                                tenant_id=tenant_id,
                                user_id=user_id,
                                idempotency_key=idempotency_key or str(uuid.uuid4()),
                                query_text=query_text,
                                mode="metadata_first",
                                k_final=final_limit,
                                k_vector=0,
                                k_bm25=0,
                                results_count=len(metadata_first_results),
                                context_texts=context_texts,
                                latency_ms=elapsed_ms,
                            )
                        except Exception:
                            logger.warning(
                                "retrieval.usage_recording_failed_metadata_first tenant_id=%s user_id=%d",
                                tenant_id,
                                user_id,
                                exc_info=True,
                            )

                    logger.info(
                        "retrieval.done tenant_id=%s user_id=%d mode=metadata_first final_hits=%d elapsed_ms=%d",
                        tenant_id,
                        user_id,
                        len(metadata_first_results),
                        elapsed_ms,
                    )
                    return metadata_first_results

                logger.info(
                    "retrieval.metadata_first_fallback tenant_id=%s user_id=%d candidate_count=%d",
                    tenant_id,
                    user_id,
                    len(metadata_first_candidates),
                )

            except Exception:
                logger.warning(
                    "retrieval.metadata_first_failed tenant_id=%s user_id=%d",
                    tenant_id,
                    user_id,
                    exc_info=True,
                )

        # ─────────────────────────────
        # Phase 2: Access control (CRITICAL — fail-closed)
        # ─────────────────────────────
        try:
            allowed = await self._access_policy.allowed_documents(
                tenant_id=tenant_id,
                user_id=user_id,
                scope=scope,
            )
        except Exception:
            logger.error(
                "retrieval.access_policy_failed tenant_id=%s user_id=%d",
                tenant_id,
                user_id,
                exc_info=True,
            )
            return []

        if not allowed:
            logger.info(
                "retrieval.no_accessible_documents tenant_id=%s user_id=%d",
                tenant_id,
                user_id,
            )
            return []

        # ─────────────────────────────
        # Phase 3: Build execution context
        # ─────────────────────────────
        ctx = await self._build_execution_context(
            query_text=query_text,
            mode=mode_norm,
            include_debug=include_debug,
            history=history,
            tenant_id=tenant_id,
            user_id=user_id,
            allowed_doc_ids=allowed,
        )

        if not ctx.candidate_doc_ids:
            logger.info(
                "retrieval.no_candidate_documents tenant_id=%s user_id=%d",
                tenant_id,
                user_id,
            )
            return []

        # ─────────────────────────────
        # Phase 4: Multi-query retrieval
        # ─────────────────────────────
        merged, vector_hits_count, bm25_hits_count = (
            await self._execute_retrieval(
                ctx=ctx,
                tenant_id=tenant_id,
                user_id=user_id,
                vector_limit=vector_limit,
                bm25_limit=bm25_limit,
                final_limit=final_limit,
            )
        )

        # ─────────────────────────────
        # Phase 5: Re-rank (fail-open)
        # ─────────────────────────────
        try:
            reranked = (
                await self._reranker.rerank(
                    query=query_text,
                    chunks=merged,
                    top_k=final_limit,
                )
                or []
            )
        except Exception:
            logger.error(
                "retrieval.rerank_failed tenant_id=%s user_id=%d",
                tenant_id,
                user_id,
                exc_info=True,
            )
            reranked = merged[:final_limit]

        # ─────────────────────────────
        # Phase 6: Intent preference resolution
        # ─────────────────────────────
        ctx = self._resolve_intent_preferences(
            ctx,
            query_text=query_text,
            tenant_id=tenant_id,
            user_id=user_id,
        )

        # ─────────────────────────────
        # Phase 7: Metadata bias application
        # ─────────────────────────────
        if (
            ctx.metadata_preference is not None
            and ctx.metadata_preference.has_preferences
            and self._metadata_bias_reranker is not None
            and reranked
        ):
            try:
                doc_ids = {c.document_id for c in reranked}
                doc_meta = await self._load_document_metadata_for_bias(
                    doc_ids, tenant_id,
                )
                reranked = self._metadata_bias_reranker.apply_bias(
                    chunks=reranked,
                    preference=ctx.metadata_preference,
                    doc_metadata=doc_meta,
                )
            except Exception:
                logger.warning(
                    "retrieval.metadata_bias_failed tenant_id=%s user_id=%d",
                    tenant_id,
                    user_id,
                    exc_info=True,
                )

        # ─────────────────────────────
        # Phase 8: Family consolidation
        # ─────────────────────────────
        selection_meta: dict[int, dict] | None = None
        source_doc_map: dict[int, int] | None = None

        if self._representation_selector is not None and reranked:
            try:
                reranked, selection_meta, source_doc_map = (
                    await self._consolidate_families(
                        query_text,
                        reranked,
                        tenant_id,
                        user_id,
                        include_debug=ctx.include_debug,
                        representation_preference=ctx.representation_preference,
                    )
                )
            except Exception:
                logger.warning(
                    "retrieval.family_consolidation_failed tenant_id=%s user_id=%d",
                    tenant_id,
                    user_id,
                    exc_info=True,
                )

        # ─────────────────────────────
        # Phase 8.5: Defense-in-depth ACL sweep (Phase 6)
        # ─────────────────────────────
        if reranked:
            try:
                reranked = await self._apply_acl_to_reranked(
                    reranked,
                    tenant_id=tenant_id,
                    acl_user_ctx=acl_user_ctx,
                )
            except Exception:
                logger.error(
                    "retrieval.acl_defense_in_depth_failed "
                    "tenant_id=%s user_id=%s — fail-closed, clearing results",
                    tenant_id,
                    acl_user_ctx.user_id if acl_user_ctx else user_id,
                    exc_info=True,
                )
                reranked = []

        # ─────────────────────────────
        # Phase 9: Response build + usage recording + telemetry
        # ─────────────────────────────
        try:
            results = self._response_builder.build(
                reranked,
                query_text,
                selection_meta=selection_meta if ctx.include_debug else None,
                source_doc_map=source_doc_map,
            )
        except Exception:
            logger.error(
                "retrieval.response_builder_failed tenant_id=%s user_id=%d",
                tenant_id,
                user_id,
                exc_info=True,
            )
            results = []

        elapsed_ms = int((time.monotonic() - start) * 1000)

        await self._record_usage_and_log(
            ctx=ctx,
            tenant_id=tenant_id,
            user_id=user_id,
            idempotency_key=idempotency_key,
            query_text=query_text,
            vector_limit=vector_limit,
            bm25_limit=bm25_limit,
            final_limit=final_limit,
            reranked=reranked,
            results=results,
            vector_hits_count=vector_hits_count,
            bm25_hits_count=bm25_hits_count,
            elapsed_ms=elapsed_ms,
        )

        return results

    # ─────────────────────────────────────────────────────────────
    # Phase 3: Execution context builders
    # ─────────────────────────────────────────────────────────────

    async def _build_execution_context(
        self,
        *,
        query_text: str,
        mode: str,
        include_debug: bool,
        history: list | None,
        tenant_id: str,
        user_id: int,
        allowed_doc_ids: set[int],
    ) -> RetrievalExecutionContext:
        """Build the pre-retrieval execution context.

        Captures:
          - Normalized mode
          - Query rewrite result (fail-open)
          - Query plan (fail-open)
          - Effective query list
          - Candidate document IDs

        All I/O and LLM calls in this method are fail-open.
        """
        mode_norm = "hybrid"
        if isinstance(mode, str):
            m = mode.strip().lower()
            if m in ("hybrid", "vector", "bm25"):
                mode_norm = m
            else:
                logger.warning(
                    "retrieval.invalid_mode tenant_id=%s user_id=%d mode=%s",
                    tenant_id,
                    user_id,
                    m,
                )

        rewrite_plan = None
        if self._query_rewriter is not None and self._query_rewriter.enabled:
            try:
                rewrite_plan = await self._query_rewriter.maybe_rewrite(
                    query_text,
                    history=history,
                )
            except Exception:
                logger.warning(
                    "retrieval.query_rewrite_failed tenant_id=%s user_id=%d",
                    tenant_id,
                    user_id,
                    exc_info=True,
                )
                rewrite_plan = None

        rewrite_usable = (
            rewrite_plan is not None and not rewrite_plan.fallback_used
        )

        if rewrite_usable:
            # ── Planner-lite: extract narrowing filters only ──────────
            # Rewrite stays authoritative for query wording / effective_queries.
            # Planner is called only to extract doc-scope narrowing (filters.doc_ids).
            # Planner subqueries / normalized_query / preferred_mode are IGNORED.
            # Fail-open: planner error → no narrowing (full allowed scope).
            narrowing_filters = QueryPlanFilters()
            if self._planner is not None and self._planner.enabled:
                try:
                    narrowing_plan = await self._planner.build_plan(
                        tenant_id=tenant_id,
                        query_text=query_text,
                    )
                    narrowing_filters = QueryPlanFilters(
                        doc_ids=narrowing_plan.filters.doc_ids,
                    )
                    logger.info(
                        "retrieval.planner_lite rewrite_usable=true "
                        "tenant_id=%s narrowing_doc_ids=%d",
                        tenant_id,
                        len(narrowing_filters.doc_ids),
                    )
                except Exception:
                    logger.warning(
                        "retrieval.planner_lite_failed rewrite_usable=true "
                        "tenant_id=%s action=no_narrowing",
                        tenant_id,
                        exc_info=True,
                    )

            plan = QueryPlan.fallback(query_text)
            # Override fallback filters with planner-lite narrowing
            plan = plan.model_copy(update={"filters": narrowing_filters})

        elif self._planner is not None and self._planner.enabled:
            try:
                plan = await self._planner.build_plan(
                    tenant_id=tenant_id,
                    query_text=query_text,
                )
            except Exception:
                plan = QueryPlan.fallback(query_text)
        else:
            plan = QueryPlan.fallback(query_text)

        if plan.filters.doc_ids:
            candidate_doc_ids = frozenset(
                allowed_doc_ids.intersection(set(plan.filters.doc_ids))
            )
        else:
            candidate_doc_ids = frozenset(allowed_doc_ids)

        if rewrite_usable:
            queries_to_run = tuple(rewrite_plan.effective_queries())
            logger.info(
                "retrieval.using_rewrite_plan tenant_id=%s mode=%s queries=%d",
                tenant_id,
                rewrite_plan.query_mode.value,
                len(queries_to_run),
            )
        else:
            queries_to_run = tuple(plan.subqueries or [plan.normalized_query])

        return RetrievalExecutionContext(
            original_query=query_text,
            effective_mode=mode_norm,
            include_debug=include_debug,
            rewrite_plan=rewrite_plan,
            rewrite_usable=rewrite_usable,
            query_plan=plan,
            effective_queries=queries_to_run,
            candidate_doc_ids=candidate_doc_ids,
            history_provided=bool(history),
        )

    def _resolve_intent_preferences(
        self,
        ctx: RetrievalExecutionContext,
        *,
        query_text: str,
        tenant_id: str,
        user_id: int,
    ) -> RetrievalExecutionContext:
        """Resolve metadata and representation intent preferences.

        Populates ctx.metadata_preference and ctx.representation_preference.
        Both are fail-open: errors result in None (no preference).

        Returns a NEW context (frozen dataclass — cannot mutate).
        """
        metadata_pref = None
        repr_pref = None

        if (
            self._metadata_intent_service is not None
            and self._metadata_intent_service.enabled
        ):
            try:
                metadata_pref = self._metadata_intent_service.parse(
                    query_text,
                    rewrite_plan=ctx.rewrite_plan,
                )
            except Exception:
                logger.warning(
                    "retrieval.metadata_intent_failed tenant_id=%s user_id=%d",
                    tenant_id,
                    user_id,
                    exc_info=True,
                )

        if (
            self._representation_intent_service is not None
            and self._representation_intent_service.enabled
        ):
            try:
                repr_pref = self._representation_intent_service.classify(
                    query_text,
                )
            except Exception:
                logger.warning(
                    "retrieval.representation_intent_failed tenant_id=%s user_id=%d",
                    tenant_id,
                    user_id,
                    exc_info=True,
                )

        if metadata_pref is not None or repr_pref is not None:
            from dataclasses import replace
            return replace(
                ctx,
                metadata_preference=metadata_pref,
                representation_preference=repr_pref,
            )
        return ctx

    # ─────────────────────────────────────────────────────────────
    # Phase 4: Retrieval execution
    # ─────────────────────────────────────────────────────────────

    async def _execute_retrieval(
        self,
        *,
        ctx: RetrievalExecutionContext,
        tenant_id: str,
        user_id: int,
        vector_limit: int,
        bm25_limit: int,
        final_limit: int,
    ) -> tuple[list[ScoredChunk], int, int]:
        """Execute multi-query retrieval using the execution context.

        Runs BM25 / Vector / Hybrid search for each effective query,
        merges results, and deduplicates by chunk_id (keep best score).

        Returns:
            (merged_chunks, vector_hits_count, bm25_hits_count)

        Embedding failure handling:
          - vector mode: HARD FAIL (returns empty) — caller's explicit contract.
          - hybrid mode: degrade to BM25-only for the affected query.
        Individual vector/BM25 search failures are fail-open.
        """
        per_q_limit = min(10, final_limit)

        vector_hits_count = 0
        bm25_hits_count = 0
        merged_all: list[ScoredChunk] = []

        for q in ctx.effective_queries:
            bm25_results: list[ScoredChunk] = []
            vector_results: list[ScoredChunk] = []

            if ctx.effective_mode in ("hybrid", "vector"):
                vector_disabled_for_query = False

                try:
                    embeddings = await self._embedding_provider.embed([q])
                    if not embeddings:
                        if ctx.effective_mode == "vector":
                            logger.error(
                                "retrieval.embedding_empty tenant_id=%s user_id=%d mode=vector",
                                tenant_id,
                                user_id,
                            )
                            return [], 0, 0

                        # hybrid: degrade to BM25-only
                        logger.warning(
                            "retrieval.embedding_empty_hybrid_degrade "
                            "tenant_id=%s user_id=%d "
                            "action=skip_vector_branch_bm25_only",
                            tenant_id,
                            user_id,
                        )
                        vector_disabled_for_query = True
                    else:
                        query_embedding = embeddings[0]
                except Exception:
                    if ctx.effective_mode == "vector":
                        logger.error(
                            "retrieval.embedding_failed tenant_id=%s user_id=%d mode=vector",
                            tenant_id,
                            user_id,
                            exc_info=True,
                        )
                        return [], 0, 0

                    # hybrid: degrade to BM25-only
                    logger.warning(
                        "retrieval.embedding_failed_hybrid_degrade "
                        "tenant_id=%s user_id=%d "
                        "action=skip_vector_branch_bm25_only",
                        tenant_id,
                        user_id,
                        exc_info=True,
                    )
                    vector_disabled_for_query = True

                if not vector_disabled_for_query:
                    try:
                        vector_results = (
                            await self._vector_retriever.search(
                                tenant_id=tenant_id,
                                query_embedding=query_embedding,
                                limit=min(vector_limit, per_q_limit),
                                filters=VectorFilter(
                                    document_ids=frozenset(ctx.candidate_doc_ids)
                                ),
                            )
                            or []
                        )
                    except Exception:
                        logger.error(
                            "retrieval.vector_failed tenant_id=%s user_id=%d",
                            tenant_id,
                            user_id,
                            exc_info=True,
                        )
                        vector_results = []

            if ctx.effective_mode in ("hybrid", "bm25"):
                try:
                    bm25_results = (
                        await self._bm25_retriever.search(
                            tenant_id=tenant_id,
                            query_text=q,
                            limit=min(bm25_limit, per_q_limit),
                            allowed_doc_ids=ctx.candidate_doc_ids,
                        )
                        or []
                    )
                except Exception:
                    logger.error(
                        "retrieval.bm25_failed tenant_id=%s user_id=%d",
                        tenant_id,
                        user_id,
                        exc_info=True,
                    )
                    bm25_results = []

            vector_hits_count += len(vector_results)
            bm25_hits_count += len(bm25_results)

            if ctx.effective_mode == "vector":
                merged_q = vector_results
            elif ctx.effective_mode == "bm25":
                merged_q = bm25_results
            else:
                try:
                    merged_q = self._hybrid.merge(
                        vector_results=vector_results,
                        bm25_results=bm25_results,
                    )
                except Exception:
                    logger.error(
                        "retrieval.hybrid_merge_failed tenant_id=%s user_id=%d",
                        tenant_id,
                        user_id,
                        exc_info=True,
                    )
                    merged_q = []

            merged_all.extend(merged_q)

        merged = self._merge_keep_best(merged_all)
        return merged[: max(final_limit, 1)], vector_hits_count, bm25_hits_count

    async def _record_usage_and_log(
        self,
        *,
        ctx: RetrievalExecutionContext,
        tenant_id: str,
        user_id: int,
        idempotency_key: str | None,
        query_text: str,
        vector_limit: int,
        bm25_limit: int,
        final_limit: int,
        reranked: list[ScoredChunk],
        results: list[QueryResult],
        vector_hits_count: int,
        bm25_hits_count: int,
        elapsed_ms: int,
    ) -> None:
        """Record query usage and emit final telemetry log.

        Non-critical only; fail-open.
        """
        if self._query_usage_service is not None:
            try:
                max_len = _safe_int(
                    getattr(settings, "QUERY_USAGE_SNIPPET_MAX_LEN", 1000),
                    1000,
                )

                context_texts = [
                    (r.snippet[:max_len] if getattr(r, "snippet", None) else "")
                    for r in reranked
                ]

                await self._query_usage_service.record_query_usage(
                    tenant_id=tenant_id,
                    user_id=user_id,
                    idempotency_key=idempotency_key or str(uuid.uuid4()),
                    query_text=query_text,
                    mode=ctx.effective_mode,
                    k_final=final_limit,
                    k_vector=vector_limit,
                    k_bm25=bm25_limit,
                    results_count=len(results),
                    context_texts=context_texts,
                    latency_ms=elapsed_ms,
                )
            except Exception:
                logger.warning(
                    "retrieval.usage_recording_failed tenant_id=%s user_id=%d",
                    tenant_id,
                    user_id,
                    exc_info=True,
                )

        rewrite_plan = getattr(ctx, "rewrite_plan", None)
        metadata_pref = getattr(ctx, "metadata_preference", None)
        repr_pref = getattr(ctx, "representation_preference", None)

        logger.info(
            "retrieval.done tenant_id=%s user_id=%d mode=%s "
            "bm25_hits=%d vector_hits=%d final_hits=%d elapsed_ms=%d "
            "rewrite=%s metadata=%s repr_policy=%s",
            tenant_id,
            user_id,
            ctx.effective_mode,
            bm25_hits_count,
            vector_hits_count,
            len(results),
            elapsed_ms,
            rewrite_plan.telemetry_dict() if rewrite_plan else "disabled",
            metadata_pref.telemetry_dict() if metadata_pref else "disabled",
            repr_pref.telemetry_dict() if repr_pref else "disabled",
        )

    # ─────────────────────────────────────────────────────────────
    # Document-level ACL helpers (Phase 6)
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _apply_acl_to_metadata_candidates(
        candidates: list,
        *,
        acl_user_ctx: UserAccessContext,
    ) -> list:
        """Filter metadata-first candidates by document-level ACL.

        Runs BEFORE is_good_enough() to prevent restricted docs from
        triggering early-return and bypassing the normal access policy.

        Receives a fully-resolved UserAccessContext with role_codes and
        permissions from system context (when available).  If system
        context resolution failed, role/permission-restricted docs will
        be denied (fail-safe).
        """
        if not candidates:
            return candidates

        filtered, trace = filter_candidates_by_acl(
            candidates,
            user_ctx=acl_user_ctx,
            get_metadata=lambda c: c.metadata if hasattr(c, "metadata") else {},
            label="metadata_first",
        )

        return filtered

    async def _apply_acl_to_reranked(
        self,
        reranked: list[ScoredChunk],
        *,
        tenant_id: str,
        acl_user_ctx: UserAccessContext,
    ) -> list[ScoredChunk]:
        """Defense-in-depth ACL sweep on reranked chunks.

        Loads document metadata for surviving chunks, extracts
        access_scope, and removes chunks from denied documents.

        This catches any restricted docs that slipped through earlier
        stages due to bugs or missing ACL checks.
        """
        if not reranked:
            return reranked

        doc_ids = {c.document_id for c in reranked}
        doc_meta = await self._load_document_metadata_for_bias(
            doc_ids, tenant_id,
        )

        allowed_doc_ids: set[int] = set()
        denied_count = 0

        for doc_id in doc_ids:
            meta = doc_meta.get(doc_id, {})
            full_meta = meta.get("meta", {})
            raw_scope, scope_present = extract_access_scope_from_metadata(full_meta)
            scope = normalize_access_scope(raw_scope, scope_present=scope_present)
            decision = evaluate_document_access(scope, acl_user_ctx)

            if decision.allowed:
                allowed_doc_ids.add(doc_id)
            else:
                denied_count += 1

        if denied_count == 0:
            return reranked

        filtered = [c for c in reranked if c.document_id in allowed_doc_ids]

        logger.info(
            "retrieval.acl_defense_in_depth "
            "tenant_id=%s user_id=%s total_docs=%d denied_docs=%d "
            "chunks_before=%d chunks_after=%d",
            tenant_id,
            acl_user_ctx.user_id,
            len(doc_ids),
            denied_count,
            len(reranked),
            len(filtered),
        )

        return filtered

    # ─────────────────────────────────────────────────────────────
    # Metadata bias helpers (Phase 3B)
    # ─────────────────────────────────────────────────────────────

    async def _load_document_metadata_for_bias(
        self, doc_ids: set[int], tenant_id: str,
    ) -> dict[int, dict]:
        """
        Batch-load document metadata for metadata bias scoring.

        Returns {doc_id: {source, title, representation_type, meta, created_at}}.
        Tenant-scoped for isolation.
        """
        from sqlalchemy import select
        from app.db.models.document import Document

        stmt = select(
            Document.id,
            Document.source,
            Document.title,
            Document.representation_type,
            Document.meta,
            Document.created_at,
        ).where(
            Document.id.in_(doc_ids),
            Document.tenant_id == tenant_id,
        )

        async with self._session_factory() as db:
            result = await db.execute(stmt)
            rows = result.fetchall()

        return {
            row[0]: {
                "source": row[1],
                "title": row[2],
                "representation_type": row[3],
                "meta": row[4] or {},
                "created_at": row[5],
            }
            for row in rows
        }

    # ─────────────────────────────────────────────────────────────
    # Family consolidation helpers (Step 4 + Step 5)
    # ─────────────────────────────────────────────────────────────

    async def _load_family_metadata(
        self, doc_ids: set[int], tenant_id: str = "",
    ) -> dict[int, tuple[str, int | None]]:
        """
        Batch-load (representation_type, parent_document_id) for given doc IDs.

        Returns {doc_id: (representation_type, parent_document_id)}.
        Documents not found default to ("original", None).

        Phase 1 hotfix: adds tenant_id filter for defense-in-depth.
        """
        from sqlalchemy import select
        from app.db.models.document import Document

        stmt = select(
            Document.id,
            Document.representation_type,
            Document.parent_document_id,
        ).where(Document.id.in_(doc_ids))

        if tenant_id:
            stmt = stmt.where(Document.tenant_id == tenant_id)

        async with self._session_factory() as db:
            result = await db.execute(stmt)
            rows = result.fetchall()

        return {
            row[0]: (row[1] or "original", row[2])
            for row in rows
        }

    async def _consolidate_families(
        self,
        query_text: str,
        reranked: list[ScoredChunk],
        tenant_id: str,
        user_id: int,
        *,
        include_debug: bool = False,
        representation_preference=None,
    ) -> tuple[
        list[ScoredChunk],
        dict[int, dict] | None,
        dict[int, int] | None,
    ]:
        """
        Run family-based deduplication on reranked chunks.

        Returns:
            (consolidated_chunks, selection_meta, source_doc_map)

            - selection_meta: {doc_id: debug_dict} (None if debug off)
            - source_doc_map: {doc_id: source_document_id}
        """
        assert self._representation_selector is not None

        doc_ids = {c.document_id for c in reranked}
        family_meta = await self._load_family_metadata(doc_ids, tenant_id)

        candidates: list[RetrievalCandidate] = []
        for c in reranked:
            repr_type, parent_id = family_meta.get(
                c.document_id, ("original", None),
            )
            candidates.append(
                RetrievalCandidate(
                    document_id=c.document_id,
                    chunk_id=c.chunk_id,
                    chunk_index=c.chunk_index,
                    score=c.score,
                    snippet=c.snippet,
                    title=c.title,
                    version_id=c.version_id,
                    source=c.source,
                    representation_type=repr_type,
                    parent_document_id=parent_id,
                )
            )

        selected = self._representation_selector.consolidate(
            query_text,
            candidates,
            representation_preference=representation_preference,
        )

        consolidated: list[ScoredChunk] = []
        source_doc_map: dict[int, int] = {}
        selection_meta: dict[int, dict] | None = {} if include_debug else None

        for s in selected:
            consolidated.append(
                ScoredChunk(
                    chunk_id=s.chunk_id,
                    document_id=s.document_id,
                    version_id=s.version_id,
                    chunk_index=s.chunk_index,
                    score=s.score,
                    source=s.source,
                    snippet=s.snippet,
                    title=s.title,
                )
            )

            source_doc_map[s.document_id] = s.source_document_id

            if selection_meta is not None:
                selection_meta[s.document_id] = {
                    "representation_type": s.selected_representation_type,
                    "parent_document_id": (
                        s.family_key
                        if s.selected_representation_type == "synthesized"
                        else None
                    ),
                    "family_key": s.family_key,
                    "selection_reason": s.selection_reason,
                    "source_document_id": s.source_document_id,
                }

        intent_val = (
            self._representation_selector.detect_intent(query_text).value
        )

        logger.info(
            "retrieval.family_consolidation tenant_id=%s user_id=%d "
            "intent=%s mode=%s before=%d after=%d families=%d",
            tenant_id,
            user_id,
            intent_val,
            self._representation_selector.mode,
            len(reranked),
            len(consolidated),
            len({s.family_key for s in selected}),
        )

        for s in selected:
            logger.debug(
                "retrieval.selected "
                "document_id=%d representation_type=%s "
                "parent_document_id=%s score=%.4f "
                "selection_reason=%s family_key=%d "
                "source_document_id=%d",
                s.document_id,
                s.selected_representation_type,
                s.family_key if s.selected_representation_type == "synthesized" else "null",
                s.score,
                s.selection_reason,
                s.family_key,
                s.source_document_id,
            )

        return consolidated, selection_meta, source_doc_map

#==================================
# from __future__ import annotations

# import logging
# import time
# import uuid
# from typing import Literal

# from app.core.config import settings
# from app.services.embedding_provider import EmbeddingProvider
# from app.services.retrieval.access_policy import AccessPolicy
# from app.services.retrieval.bm25_retriever import BM25Retriever
# from app.services.retrieval.hybrid import HybridConfig, HybridStrategy
# from app.services.retrieval.query_plan import QueryPlan
# from app.services.retrieval.query_planner import QueryPlanner
# from app.services.retrieval.reranker import DeterministicReRanker, ReRanker
# from app.services.retrieval.response_builder import ResponseBuilder
# from app.services.retrieval.types import QueryResult, QueryScope, ScoredChunk, VectorFilter
# from app.services.retrieval.vector_retriever import VectorRetriever
# from app.services.retrieval.document_representation_selector import (
#     DocumentRepresentationSelector,
#     RetrievalCandidate,
#     SelectedCandidate,
# )

# logger = logging.getLogger(__name__)

# QueryMode = Literal["hybrid", "vector", "bm25"]


# def _safe_int(value, default: int) -> int:
#     try:
#         v = int(value)
#         return v if v > 0 else default
#     except Exception:
#         return default


# class QueryService:
#     """
#     Orchestrates the full retrieval pipeline.
#     All dependencies are injected — vendor logic stays in adapters.
#     """

#     __slots__ = (
#         "_access_policy",
#         "_embedding_provider",
#         "_vector_retriever",
#         "_bm25_retriever",
#         "_hybrid",
#         "_reranker",
#         "_response_builder",
#         "_query_usage_service",
#         "_planner",
#         "_representation_selector",
#         "_session_factory",
#         "_query_rewriter",
#         "_metadata_intent_service",
#         "_metadata_bias_reranker",
#         "_representation_intent_service",
#     )

#     def __init__(
#         self,
#         *,
#         access_policy: AccessPolicy,
#         embedding_provider: EmbeddingProvider,
#         vector_retriever: VectorRetriever,
#         bm25_retriever: BM25Retriever,
#         hybrid_config: HybridConfig | None = None,
#         reranker: ReRanker | None = None,
#         response_builder: ResponseBuilder | None = None,
#         query_usage_service=None,
#         planner: QueryPlanner | None = None,
#         representation_selector: DocumentRepresentationSelector | None = None,
#         session_factory=None,
#         query_rewriter=None,
#         metadata_intent_service=None,
#         metadata_bias_reranker=None,
#         representation_intent_service=None,
#     ) -> None:
#         self._access_policy = access_policy
#         self._embedding_provider = embedding_provider
#         self._vector_retriever = vector_retriever
#         self._bm25_retriever = bm25_retriever
#         self._hybrid = HybridStrategy(hybrid_config)
#         self._reranker: ReRanker = reranker or DeterministicReRanker()
#         self._response_builder = response_builder or ResponseBuilder()
#         self._query_usage_service = query_usage_service
#         self._planner = planner
#         self._representation_selector = representation_selector
#         self._query_rewriter = query_rewriter
#         self._metadata_intent_service = metadata_intent_service
#         self._metadata_bias_reranker = metadata_bias_reranker
#         self._representation_intent_service = representation_intent_service
#         if session_factory is None:
#             from app.db.session import AsyncSessionLocal
#             session_factory = AsyncSessionLocal
#         self._session_factory = session_factory

#     @staticmethod
#     def _merge_keep_best(chunks: list[ScoredChunk]) -> list[ScoredChunk]:
#         by_chunk_id: dict[int, ScoredChunk] = {}
#         for chunk in chunks:
#             prev = by_chunk_id.get(chunk.chunk_id)
#             if prev is None or chunk.score > prev.score:
#                 by_chunk_id[chunk.chunk_id] = chunk
#         merged = list(by_chunk_id.values())
#         merged.sort(key=lambda c: (-c.score, c.chunk_id))
#         return merged

#     async def query(
#         self,
#         *,
#         tenant_id: str,
#         user_id: int,
#         query_text: str,
#         scope: QueryScope | None = None,
#         vector_limit: int | None = None,
#         bm25_limit: int | None = None,
#         final_limit: int | None = None,
#         idempotency_key: str | None = None,
#         mode: QueryMode = "hybrid",
#         include_debug: bool = False,
#         history: list | None = None,
#     ) -> list[QueryResult]:

#         start = time.monotonic()

#         scope = scope or QueryScope()
#         vector_limit = _safe_int(
#             vector_limit, getattr(settings, "QUERY_VECTOR_LIMIT", 30)
#         )
#         bm25_limit = _safe_int(
#             bm25_limit, getattr(settings, "QUERY_BM25_LIMIT", 30)
#         )
#         final_limit = _safe_int(
#             final_limit, getattr(settings, "QUERY_FINAL_LIMIT", 10)
#         )

#         # ─────────────────────────────
#         # Normalize mode defensively
#         # ─────────────────────────────
#         mode_norm = "hybrid"
#         if isinstance(mode, str):
#             m = mode.strip().lower()
#             if m in ("hybrid", "vector", "bm25"):
#                 mode_norm = m
#             else:
#                 logger.warning(
#                     "retrieval.invalid_mode tenant_id=%s user_id=%d mode=%s",
#                     tenant_id,
#                     user_id,
#                     m,
#                 )

#         # ─────────────────────────────
#         # 1) Access Control (CRITICAL) — fail-closed
#         # ─────────────────────────────
#         try:
#             allowed = await self._access_policy.allowed_documents(
#                 tenant_id=tenant_id,
#                 user_id=user_id,
#                 scope=scope,
#             )
#         except Exception:
#             logger.error(
#                 "retrieval.access_policy_failed tenant_id=%s user_id=%d",
#                 tenant_id,
#                 user_id,
#                 exc_info=True,
#             )
#             return []

#         if not allowed:
#             logger.info(
#                 "retrieval.no_accessible_documents tenant_id=%s user_id=%d",
#                 tenant_id,
#                 user_id,
#             )
#             return []

#         # ─────────────────────────────
#         # 2a) Query Rewrite (Phase 3A — fail-open)
#         # ─────────────────────────────
#         rewrite_plan = None
#         if self._query_rewriter is not None and self._query_rewriter.enabled:
#             try:
#                 rewrite_plan = await self._query_rewriter.maybe_rewrite(
#                     query_text, history=history,
#                 )
#             except Exception:
#                 logger.warning(
#                     "retrieval.query_rewrite_failed tenant_id=%s user_id=%d",
#                     tenant_id, user_id, exc_info=True,
#                 )
#                 rewrite_plan = None

#         rewrite_usable = (
#             rewrite_plan is not None and not rewrite_plan.fallback_used
#         )

#         # ─────────────────────────────
#         # 2b) Query plan — only when rewrite NOT usable
#         # ─────────────────────────────
#         # When rewrite is usable, skip planner to save latency/tokens.
#         # Still need a lightweight plan for doc_id filtering.
#         if rewrite_usable:
#             plan = QueryPlan.fallback(query_text)
#             logger.info(
#                 "retrieval.planner_skipped rewrite_usable=true tenant_id=%s",
#                 tenant_id,
#             )
#         elif self._planner is not None and self._planner.enabled:
#             try:
#                 plan = await self._planner.build_plan(tenant_id=tenant_id, query_text=query_text)
#             except Exception:
#                 plan = QueryPlan.fallback(query_text)
#         else:
#             plan = QueryPlan.fallback(query_text)

#         if plan.filters.doc_ids:
#             candidate_doc_ids = allowed.intersection(set(plan.filters.doc_ids))
#         else:
#             candidate_doc_ids = set(allowed)

#         if not candidate_doc_ids:
#             logger.info(
#                 "retrieval.no_candidate_documents tenant_id=%s user_id=%d",
#                 tenant_id,
#                 user_id,
#             )
#             return []

#         # ─────────────────────────────
#         # 2c) Build effective query list
#         # ─────────────────────────────
#         if rewrite_usable:
#             queries_to_run = rewrite_plan.effective_queries()
#             logger.info(
#                 "retrieval.using_rewrite_plan tenant_id=%s mode=%s queries=%d",
#                 tenant_id, rewrite_plan.query_mode.value, len(queries_to_run),
#             )
#         else:
#             queries_to_run = plan.subqueries or [plan.normalized_query]

#         per_q_limit = min(10, final_limit)

#         vector_hits_count = 0
#         bm25_hits_count = 0
#         merged_all: list[ScoredChunk] = []

#         # ─────────────────────────────
#         # 3) Mode-specific retrieval per query
#         # ─────────────────────────────
#         for q in queries_to_run:
#             bm25_results: list[ScoredChunk] = []
#             vector_results: list[ScoredChunk] = []

#             if mode_norm in ("hybrid", "vector"):
#                 try:
#                     embeddings = await self._embedding_provider.embed([q])
#                     if not embeddings:
#                         logger.error(
#                             "retrieval.embedding_empty tenant_id=%s user_id=%d",
#                             tenant_id,
#                             user_id,
#                         )
#                         return []
#                     query_embedding = embeddings[0]
#                 except Exception:
#                     logger.error(
#                         "retrieval.embedding_failed tenant_id=%s user_id=%d",
#                         tenant_id,
#                         user_id,
#                         exc_info=True,
#                     )
#                     return []

#                 try:
#                     vector_results = (
#                         await self._vector_retriever.search(
#                             tenant_id=tenant_id,
#                             query_embedding=query_embedding,
#                             limit=min(vector_limit, per_q_limit),
#                             filters=VectorFilter(
#                                 document_ids=frozenset(candidate_doc_ids)
#                             ),
#                         )
#                         or []
#                     )
#                 except Exception:
#                     logger.error(
#                         "retrieval.vector_failed tenant_id=%s user_id=%d",
#                         tenant_id,
#                         user_id,
#                         exc_info=True,
#                     )
#                     vector_results = []

#             if mode_norm in ("hybrid", "bm25"):
#                 try:
#                     bm25_results = (
#                         await self._bm25_retriever.search(
#                             tenant_id=tenant_id,
#                             query_text=q,
#                             limit=min(bm25_limit, per_q_limit),
#                             allowed_doc_ids=candidate_doc_ids,
#                         )
#                         or []
#                     )
#                 except Exception:
#                     logger.error(
#                         "retrieval.bm25_failed tenant_id=%s user_id=%d",
#                         tenant_id,
#                         user_id,
#                         exc_info=True,
#                     )
#                     bm25_results = []

#             vector_hits_count += len(vector_results)
#             bm25_hits_count += len(bm25_results)

#             if mode_norm == "vector":
#                 merged_q = vector_results
#             elif mode_norm == "bm25":
#                 merged_q = bm25_results
#             else:
#                 try:
#                     merged_q = self._hybrid.merge(
#                         vector_results=vector_results,
#                         bm25_results=bm25_results,
#                     )
#                 except Exception:
#                     logger.error(
#                         "retrieval.hybrid_merge_failed tenant_id=%s user_id=%d",
#                         tenant_id,
#                         user_id,
#                         exc_info=True,
#                     )
#                     merged_q = []
#             merged_all.extend(merged_q)

#         merged = self._merge_keep_best(merged_all)

#         # ─────────────────────────────
#         # 4) Re-rank (fail-open)
#         # ─────────────────────────────
#         try:
#             reranked = (
#                 await self._reranker.rerank(
#                     query=query_text,
#                     chunks=merged,
#                     top_k=final_limit,
#                 )
#                 or []
#             )
#         except Exception:
#             logger.error(
#                 "retrieval.rerank_failed tenant_id=%s user_id=%d",
#                 tenant_id,
#                 user_id,
#                 exc_info=True,
#             )
#             reranked = merged[:final_limit]

#         # ─────────────────────────────
#         # 4.2) Metadata bias (Phase 3B — fail-open)
#         # ─────────────────────────────
#         metadata_pref = None
#         if (
#             self._metadata_intent_service is not None
#             and self._metadata_intent_service.enabled
#             and self._metadata_bias_reranker is not None
#             and reranked
#         ):
#             try:
#                 metadata_pref = self._metadata_intent_service.parse(
#                     query_text, rewrite_plan=rewrite_plan,
#                 )
#                 if metadata_pref and metadata_pref.has_preferences:
#                     doc_ids = {c.document_id for c in reranked}
#                     doc_meta = await self._load_document_metadata_for_bias(
#                         doc_ids, tenant_id,
#                     )
#                     reranked = self._metadata_bias_reranker.apply_bias(
#                         chunks=reranked,
#                         preference=metadata_pref,
#                         doc_metadata=doc_meta,
#                     )
#             except Exception:
#                 logger.warning(
#                     "retrieval.metadata_bias_failed tenant_id=%s user_id=%d",
#                     tenant_id, user_id, exc_info=True,
#                 )
#                 # fail-open: use base reranked results

#         # ─────────────────────────────
#         # 4.5) Representation policy (Phase 3D — fail-open)
#         # ─────────────────────────────
#         repr_pref = None
#         if (
#             self._representation_intent_service is not None
#             and self._representation_intent_service.enabled
#         ):
#             try:
#                 repr_pref = self._representation_intent_service.classify(query_text)
#             except Exception:
#                 logger.warning(
#                     "retrieval.representation_intent_failed tenant_id=%s user_id=%d",
#                     tenant_id, user_id, exc_info=True,
#                 )

#         # ─────────────────────────────
#         # 4.6) Family consolidation (fail-open)
#         # ─────────────────────────────
#         selection_meta: dict[int, dict] | None = None
#         source_doc_map: dict[int, int] | None = None

#         if self._representation_selector is not None and reranked:
#             try:
#                 reranked, selection_meta, source_doc_map = (
#                     await self._consolidate_families(
#                         query_text, reranked, tenant_id, user_id,
#                         include_debug=include_debug,
#                         representation_preference=repr_pref,
#                     )
#                 )
#             except Exception:
#                 logger.warning(
#                     "retrieval.family_consolidation_failed tenant_id=%s user_id=%d",
#                     tenant_id,
#                     user_id,
#                     exc_info=True,
#                 )
#                 # fail-open: use original reranked list, no metadata

#         # ─────────────────────────────
#         # 5) Build response (fail-open)
#         # ─────────────────────────────
#         try:
#             results = self._response_builder.build(
#                 reranked,
#                 query_text,
#                 selection_meta=selection_meta if include_debug else None,
#                 source_doc_map=source_doc_map,
#             )
#         except Exception:
#             logger.error(
#                 "retrieval.response_builder_failed tenant_id=%s user_id=%d",
#                 tenant_id,
#                 user_id,
#                 exc_info=True,
#             )
#             results = []

#         elapsed_ms = int((time.monotonic() - start) * 1000)

#         # ─────────────────────────────
#         # 6) Usage recording (non-critical) — fail-open
#         # ─────────────────────────────
#         if self._query_usage_service is not None:
#             try:
#                 max_len = _safe_int(
#                     getattr(settings, "QUERY_USAGE_SNIPPET_MAX_LEN", 1000),
#                     1000,
#                 )

#                 context_texts = [
#                     (r.snippet[:max_len] if getattr(r, "snippet", None) else "")
#                     for r in reranked
#                 ]

#                 await self._query_usage_service.record_query_usage(
#                     tenant_id=tenant_id,
#                     user_id=user_id,
#                     idempotency_key=idempotency_key or str(uuid.uuid4()),
#                     query_text=query_text,
#                     mode=mode_norm,
#                     k_final=final_limit,
#                     k_vector=vector_limit,
#                     k_bm25=bm25_limit,
#                     results_count=len(results),
#                     context_texts=context_texts,
#                     latency_ms=elapsed_ms,
#                 )
#             except Exception:
#                 logger.warning(
#                     "retrieval.usage_recording_failed tenant_id=%s user_id=%d",
#                     tenant_id,
#                     user_id,
#                     exc_info=True,
#                 )

#         # No raw text logging
#         rewrite_meta = rewrite_plan.telemetry_dict() if rewrite_plan else {}
#         metadata_meta = metadata_pref.telemetry_dict() if metadata_pref else {}
#         repr_meta = repr_pref.telemetry_dict() if repr_pref else {}
#         logger.info(
#             "retrieval.done tenant_id=%s user_id=%d mode=%s "
#             "bm25_hits=%d vector_hits=%d final_hits=%d elapsed_ms=%d "
#             "rewrite=%s metadata=%s repr_policy=%s",
#             tenant_id,
#             user_id,
#             mode_norm,
#             bm25_hits_count,
#             vector_hits_count,
#             len(results),
#             elapsed_ms,
#             rewrite_meta or "disabled",
#             metadata_meta or "disabled",
#             repr_meta or "disabled",
#         )

#         return results

#     # ─────────────────────────────────────────────────────────────
#     # Metadata bias helpers (Phase 3B)
#     # ─────────────────────────────────────────────────────────────

#     async def _load_document_metadata_for_bias(
#         self, doc_ids: set[int], tenant_id: str,
#     ) -> dict[int, dict]:
#         """
#         Batch-load document metadata for metadata bias scoring.

#         Returns {doc_id: {source, title, representation_type, meta, created_at}}.
#         Tenant-scoped for isolation.
#         """
#         from sqlalchemy import select
#         from app.db.models.document import Document

#         stmt = select(
#             Document.id,
#             Document.source,
#             Document.title,
#             Document.representation_type,
#             Document.meta,
#             Document.created_at,
#         ).where(
#             Document.id.in_(doc_ids),
#             Document.tenant_id == tenant_id,
#         )

#         async with self._session_factory() as db:
#             result = await db.execute(stmt)
#             rows = result.fetchall()

#         return {
#             row[0]: {
#                 "source": row[1],
#                 "title": row[2],
#                 "representation_type": row[3],
#                 "meta": row[4] or {},
#                 "created_at": row[5],
#             }
#             for row in rows
#         }

#     # ─────────────────────────────────────────────────────────────
#     # Family consolidation helpers (Step 4 + Step 5)
#     # ─────────────────────────────────────────────────────────────

#     async def _load_family_metadata(
#         self, doc_ids: set[int], tenant_id: str = "",
#     ) -> dict[int, tuple[str, int | None]]:
#         """
#         Batch-load (representation_type, parent_document_id) for given doc IDs.

#         Returns {doc_id: (representation_type, parent_document_id)}.
#         Documents not found default to ("original", None).

#         Phase 1 hotfix: adds tenant_id filter for defense-in-depth.
#         """
#         from sqlalchemy import select
#         from app.db.models.document import Document

#         stmt = select(
#             Document.id,
#             Document.representation_type,
#             Document.parent_document_id,
#         ).where(Document.id.in_(doc_ids))

#         if tenant_id:
#             stmt = stmt.where(Document.tenant_id == tenant_id)

#         async with self._session_factory() as db:
#             result = await db.execute(stmt)
#             rows = result.fetchall()

#         return {
#             row[0]: (row[1] or "original", row[2])
#             for row in rows
#         }

#     async def _consolidate_families(
#         self,
#         query_text: str,
#         reranked: list[ScoredChunk],
#         tenant_id: str,
#         user_id: int,
#         *,
#         include_debug: bool = False,
#         representation_preference=None,
#     ) -> tuple[
#         list[ScoredChunk],
#         dict[int, dict] | None,
#         dict[int, int] | None,
#     ]:
#         """
#         Run family-based deduplication on reranked chunks.

#         Returns:
#             (consolidated_chunks, selection_meta, source_doc_map)

#             - selection_meta: {doc_id: debug_dict} (None if debug off)
#             - source_doc_map: {doc_id: source_document_id}
#         """
#         assert self._representation_selector is not None

#         doc_ids = {c.document_id for c in reranked}
#         family_meta = await self._load_family_metadata(doc_ids, tenant_id)

#         # Build candidates
#         candidates: list[RetrievalCandidate] = []
#         for c in reranked:
#             repr_type, parent_id = family_meta.get(
#                 c.document_id, ("original", None),
#             )
#             candidates.append(
#                 RetrievalCandidate(
#                     document_id=c.document_id,
#                     chunk_id=c.chunk_id,
#                     chunk_index=c.chunk_index,
#                     score=c.score,
#                     snippet=c.snippet,
#                     title=c.title,
#                     version_id=c.version_id,
#                     source=c.source,
#                     representation_type=repr_type,
#                     parent_document_id=parent_id,
#                 )
#             )

#         selected = self._representation_selector.consolidate(
#             query_text, candidates,
#             representation_preference=representation_preference,
#         )

#         # Convert back to ScoredChunk for response builder
#         consolidated: list[ScoredChunk] = []
#         source_doc_map: dict[int, int] = {}
#         selection_meta: dict[int, dict] | None = {} if include_debug else None

#         for s in selected:
#             consolidated.append(
#                 ScoredChunk(
#                     chunk_id=s.chunk_id,
#                     document_id=s.document_id,
#                     version_id=s.version_id,
#                     chunk_index=s.chunk_index,
#                     score=s.score,
#                     source=s.source,
#                     snippet=s.snippet,
#                     title=s.title,
#                 )
#             )

#             # Source fidelity map (always)
#             source_doc_map[s.document_id] = s.source_document_id

#             # Debug metadata (only when debug active)
#             if selection_meta is not None:
#                 selection_meta[s.document_id] = {
#                     "representation_type": s.selected_representation_type,
#                     "parent_document_id": (
#                         s.family_key
#                         if s.selected_representation_type == "synthesized"
#                         else None
#                     ),
#                     "family_key": s.family_key,
#                     "selection_reason": s.selection_reason,
#                     "source_document_id": s.source_document_id,
#                 }

#         # ── Observability logging (Step 5) ────────────────────────
#         intent_val = (
#             self._representation_selector.detect_intent(query_text).value
#         )

#         logger.info(
#             "retrieval.family_consolidation tenant_id=%s user_id=%d "
#             "intent=%s mode=%s before=%d after=%d families=%d",
#             tenant_id,
#             user_id,
#             intent_val,
#             self._representation_selector.mode,
#             len(reranked),
#             len(consolidated),
#             len({s.family_key for s in selected}),
#         )

#         # Per-selection detail at DEBUG level (structured, no raw text)
#         for s in selected:
#             logger.debug(
#                 "retrieval.selected "
#                 "document_id=%d representation_type=%s "
#                 "parent_document_id=%s score=%.4f "
#                 "selection_reason=%s family_key=%d "
#                 "source_document_id=%d",
#                 s.document_id,
#                 s.selected_representation_type,
#                 s.family_key if s.selected_representation_type == "synthesized" else "null",
#                 s.score,
#                 s.selection_reason,
#                 s.family_key,
#                 s.source_document_id,
#             )

#         return consolidated, selection_meta, source_doc_map

