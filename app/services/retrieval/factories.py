"""
Factory functions for the retrieval engine.

Production-grade wiring:
- No side effects at import time
- Graceful degradation for optional backends
- Strict handling for critical components (usage / billing)
- No circular imports
"""

from __future__ import annotations

import logging

from app.core.config import settings
from app.services.embedding_provider import get_embedding_provider
from app.services.retrieval.access_policy import (
    AccessPolicy,
    DefaultTenantAccessPolicy,
)
from app.services.retrieval.bm25_repo import PgBM25Repository
from app.services.retrieval.bm25_retriever import (
    BM25Retriever,
    DefaultBM25Retriever,
    NullBM25Retriever,
)
from app.services.retrieval.hybrid import HybridConfig
from app.services.retrieval.none_planner_provider import NonePlannerProvider
from app.services.retrieval.openai_planner_provider import OpenAIPlannerProvider
from app.services.retrieval.planner_cache import PlannerCache
from app.services.retrieval.reranker import (
    DeterministicReRanker,
    ReRanker,
)
from app.services.retrieval.query_planner import QueryPlanner
from app.services.retrieval.response_builder import ResponseBuilder
from app.services.retrieval.vector_retriever import (
    DefaultVectorRetriever,
    NullVectorRetriever,
    VectorRetriever,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Core Components
# ─────────────────────────────────────────────────────────────

def get_access_policy() -> AccessPolicy:
    """
    Access policy is mandatory.
    """
    return DefaultTenantAccessPolicy()


def get_vector_retriever() -> VectorRetriever:
    """
    Build VectorRetriever according to settings.VECTOR_INDEX.

    Supported:
      - pgvector
      - qdrant
      - null
    """

    name = settings.VECTOR_INDEX.lower().strip()

    if name == "pgvector":
        from app.services.retrieval.vector_retriever import (
            PgVectorSearchBackend,
        )

        backend = PgVectorSearchBackend()
        logger.info("retrieval.factory vector_backend=pgvector")
        return DefaultVectorRetriever(backend=backend)

    if name == "qdrant":
        try:
            from app.services.retrieval.vector_retriever import (
                QdrantVectorSearchBackend,
            )

            backend = QdrantVectorSearchBackend()
            return DefaultVectorRetriever(backend=backend)

        except ImportError as exc:
            logger.warning(
                "retrieval.factory qdrant_import_error reason=%s falling_back=null",
                type(exc).__name__,
            )
            return NullVectorRetriever()

        except Exception as exc:
            logger.error(
                "retrieval.factory qdrant_runtime_error reason=%s falling_back=null",
                type(exc).__name__,
                exc_info=True,
            )
            return NullVectorRetriever()

    if name == "null":
        return NullVectorRetriever()

    logger.warning(
        "retrieval.factory vector_index=%s unsupported falling_back=null",
        name,
    )
    return NullVectorRetriever()


def get_bm25_retriever() -> BM25Retriever:
    """
    Build BM25 retriever.

    Requires database availability.
    """

    if not settings.DATABASE_URL:
        logger.warning(
            "retrieval.factory bm25_disabled no_database_url"
        )
        return NullBM25Retriever()

    try:
        repo = PgBM25Repository()
        return DefaultBM25Retriever(repo=repo)

    except Exception as exc:
        logger.error(
            "retrieval.factory bm25_init_failed reason=%s falling_back=null",
            type(exc).__name__,
            exc_info=True,
        )
        return NullBM25Retriever()


def get_reranker() -> ReRanker:
    """
    Deterministic re-ranker (safe, no external deps).
    """
    return DeterministicReRanker()


def get_hybrid_config() -> HybridConfig:
    """
    Build hybrid config from settings.
    """
    return HybridConfig(
        vector_weight=settings.HYBRID_VECTOR_WEIGHT,
        bm25_weight=settings.HYBRID_BM25_WEIGHT,
        threshold=settings.HYBRID_THRESHOLD,
    )


def get_planner_provider():
    name = (settings.LLM_QUERY_PLANNER_PROVIDER or "none").strip().lower()
    if name == "openai":
        return OpenAIPlannerProvider()
    return NonePlannerProvider()


def get_query_planner() -> QueryPlanner:
    cache = PlannerCache(ttl_s=settings.LLM_QUERY_PLANNER_CACHE_TTL_S, max_entries=1000)
    return QueryPlanner(
        provider=get_planner_provider(),
        enabled=settings.LLM_QUERY_PLANNER_ENABLED,
        cache=cache,
    )


def get_query_rewriter():
    """Build QueryRewriteService (Phase 3A — fail-open)."""
    if not getattr(settings, "QUERY_REWRITE_ENABLED", False):
        logger.info("retrieval.factory query_rewriter=disabled")
        return None

    try:
        from app.services.query_rewrite_service import QueryRewriteService
        svc = QueryRewriteService()
        logger.info(
            "retrieval.factory query_rewriter=enabled provider=%s model=%s",
            getattr(settings, "QUERY_REWRITE_PROVIDER", "none"),
            getattr(settings, "QUERY_REWRITE_MODEL", "gpt-4o-mini"),
        )
        return svc
    except Exception:
        logger.warning(
            "retrieval.factory query_rewriter_init_failed falling_back=disabled",
            exc_info=True,
        )
        return None


# ─────────────────────────────────────────────────────────────
# Query Usage / Billing (CRITICAL)
# ─────────────────────────────────────────────────────────────

def get_query_usage_service():
    """
    Wire QueryUsageService (idempotency + quota + token ledger).

    IMPORTANT:
    - ImportError → degrade gracefully
    - Runtime DB errors → RAISE (billing must not silently disable)
    """

    try:
        from app.repos.query_usage_repo import PgQueryUsageRepository
        from app.services.query_usage_service import QueryUsageService
        from app.services.token_ledger import get_token_ledger

    except ImportError as exc:
        logger.warning(
            "retrieval.factory query_usage_import_error reason=%s disabled",
            type(exc).__name__,
        )
        return None

    # Runtime errors should NOT be swallowed silently
    repo = PgQueryUsageRepository()
    ledger = get_token_ledger()

    return QueryUsageService(
        repo=repo,
        ledger=ledger,
    )


# ─────────────────────────────────────────────────────────────
# Final Query Service Factory
# ─────────────────────────────────────────────────────────────

def get_query_service():
    """
    Build a fully-wired QueryService.

    Includes:
      - Access policy
      - Embedding provider
      - Vector retriever
      - BM25 retriever
      - Hybrid strategy
      - Reranker
      - Response builder
      - Query usage service (idempotency + quota)
      - Representation selector (when synthesis enabled)
    """

    from app.services.retrieval.query_service import QueryService

    # Representation selector: only active when synthesis feature is enabled
    representation_selector = None
    try:
        if getattr(settings, "SYNTHESIS_ENABLED", False):
            from app.services.retrieval.document_representation_selector import (
                DocumentRepresentationSelector,
            )
            representation_selector = DocumentRepresentationSelector(
                mode=getattr(settings, "RETRIEVAL_REPRESENTATION_MODE", "balanced"),
            )
            logger.info(
                "retrieval.factory representation_selector=enabled mode=%s",
                representation_selector.mode,
            )
    except Exception:
        logger.warning(
            "retrieval.factory representation_selector_init_failed falling_back=disabled",
            exc_info=True,
        )

    # ── Phase 3B: Metadata-aware retrieval ────────────────────────
    metadata_intent_service = None
    metadata_bias_reranker = None
    if getattr(settings, "METADATA_RETRIEVAL_ENABLED", False):
        try:
            from app.services.retrieval.metadata_intent_service import MetadataIntentService
            from app.services.retrieval.metadata_bias import MetadataBiasReranker
            metadata_intent_service = MetadataIntentService()
            metadata_bias_reranker = MetadataBiasReranker()
            logger.info("retrieval.factory metadata_retrieval=enabled")
        except Exception:
            logger.warning(
                "retrieval.factory metadata_retrieval_init_failed falling_back=disabled",
                exc_info=True,
            )
    else:
        logger.info("retrieval.factory metadata_retrieval=disabled")

    # ── Phase 3D: Representation policy ───────────────────────────
    representation_intent_service = None
    if getattr(settings, "REPRESENTATION_POLICY_ENABLED", False):
        try:
            from app.services.retrieval.representation_intent_service import RepresentationIntentService
            representation_intent_service = RepresentationIntentService()
            logger.info("retrieval.factory representation_policy=enabled")
        except Exception:
            logger.warning(
                "retrieval.factory representation_policy_init_failed falling_back=disabled",
                exc_info=True,
            )
    else:
        logger.info("retrieval.factory representation_policy=disabled")

    return QueryService(
        access_policy=get_access_policy(),
        embedding_provider=get_embedding_provider(),
        vector_retriever=get_vector_retriever(),
        bm25_retriever=get_bm25_retriever(),
        hybrid_config=get_hybrid_config(),
        reranker=get_reranker(),
        response_builder=ResponseBuilder(),
        query_usage_service=get_query_usage_service(),
        planner=get_query_planner(),
        representation_selector=representation_selector,
        query_rewriter=get_query_rewriter(),
        metadata_intent_service=metadata_intent_service,
        metadata_bias_reranker=metadata_bias_reranker,
        representation_intent_service=representation_intent_service,
    )

