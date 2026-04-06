"""
Query usage service — billing-safe, idempotent, privacy-safe.

Responsibilities:
  1. SHA-256 hash the query text (never stored raw).
  2. Count query + context tokens via the existing tokenizer.
  3. Persist a ``QueryUsage`` record (idempotent on tenant + key).
  4. Write a billing record via ``TokenLedgerService`` (idempotent
     on a *separate* billing key so retries are safe).

NO raw query text is stored or logged anywhere in this layer.
"""
from __future__ import annotations

import hashlib
import logging
import uuid

from app.db.models.query_usage import QueryUsage
from app.nlp import get_tokenizer
from app.nlp.types import Tokenizer
from app.repos.query_usage_repo import QueryUsageRepository

logger = logging.getLogger(__name__)


class QueryUsageService:
    """
    Records query usage for billing and analytics.

    Every public method is **idempotent**: calling with the same
    ``(tenant_id, idempotency_key)`` pair multiple times produces
    exactly one ``query_usages`` row and exactly one billing entry.
    """

    __slots__ = ("_repo", "_tokenizer")

    def __init__(
        self,
        *,
        repo: QueryUsageRepository,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        self._repo = repo
        self._tokenizer = tokenizer or get_tokenizer()

    # ── public API ────────────────────────────────────────────────────

    async def record_query_usage(
        self,
        *,
        tenant_id: str,
        user_id: int | None,
        idempotency_key: str,
        query_text: str,
        mode: str,
        k_final: int,
        k_vector: int | None = None,
        k_bm25: int | None = None,
        results_count: int,
        context_texts: list[str],
        latency_ms: int,
    ) -> QueryUsage:
        """
        Record a single query for billing.

        *query_text* is used ONLY for hashing and token counting — it is
        **never** persisted or logged.

        Returns the ``QueryUsage`` record (possibly pre-existing on retry).
        """
        # 1. Privacy-safe hash
        query_hash = hashlib.sha256(query_text.encode("utf-8")).hexdigest()
        query_len = len(query_text)

        # 2. Token accounting
        tokens_query = self._tokenizer.count(query_text)
        tokens_context = sum(
            self._tokenizer.count(t) for t in context_texts
        )
        tokens_total = tokens_query + tokens_context

        # 3. Build usage record
        usage = QueryUsage(
            id=uuid.uuid4(),
            tenant_id=tenant_id,
            user_id=user_id,
            idempotency_key=idempotency_key,
            query_hash=query_hash,
            query_len=query_len,
            mode=mode,
            k_final=k_final,
            k_vector=k_vector,
            k_bm25=k_bm25,
            results_count=results_count,
            tokens_query=tokens_query,
            tokens_context=tokens_context,
            tokens_total=tokens_total,
            latency_ms=latency_ms,
        )

        # 4. Insert usage record (idempotent via unique constraint)
        record, was_inserted = await self._repo.insert_if_absent(usage=usage)

        if was_inserted:
            logger.info(
                "query_usage.recorded tenant_id=%s user_id=%s "
                "tokens_query=%d tokens_context=%d tokens_total=%d "
                "latency_ms=%d results_count=%d",
                tenant_id,
                user_id,
                tokens_query,
                tokens_context,
                tokens_total,
                latency_ms,
                results_count,
            )
        else:
            logger.info(
                "query_usage.skipped tenant_id=%s idempotency_key=%s "
                "reason=already_recorded",
                tenant_id,
                idempotency_key,
            )

        return record
