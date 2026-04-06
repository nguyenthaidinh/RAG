"""
Query usage retention service (Phase 4.3).

Deletes ``query_usages`` records older than the configured retention window.
Operates in batches, logs counts only (no content), and is safe to run
repeatedly (idempotent — second run deletes 0 if nothing new expired).

🚫 No raw query text is ever accessed or logged.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.models.query_usage import QueryUsage

logger = logging.getLogger(__name__)

# Batch size for deletion to avoid long-running transactions
_BATCH_SIZE = 1000


class QueryRetentionService:
    """
    Purges expired ``query_usages`` rows in bounded batches.

    Configuration:
      * ``settings.QUERY_USAGE_RETENTION_DAYS`` — records older than this
        many days (from *now*) are eligible for deletion.
    """

    __slots__ = ()

    @staticmethod
    def _cutoff(now: datetime) -> datetime:
        """Compute the retention cutoff timestamp."""
        days = settings.QUERY_USAGE_RETENTION_DAYS
        return now - timedelta(days=days)

    async def purge_expired_query_usages(
        self,
        db: AsyncSession,
        *,
        now: datetime | None = None,
    ) -> int:
        """
        Delete query_usages older than the retention window.

        Args:
            db: Async database session (caller owns lifecycle).
            now: Override "current time" for testing.  Defaults to UTC now.

        Returns:
            Total number of rows deleted across all batches.
        """
        if now is None:
            now = datetime.now(timezone.utc)
        elif now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        cutoff = self._cutoff(now)
        total_deleted = 0

        logger.info(
            "retention.start cutoff=%s retention_days=%d",
            cutoff.isoformat(),
            settings.QUERY_USAGE_RETENTION_DAYS,
        )

        while True:
            # Select a batch of IDs to delete (bounded scan)
            id_stmt = (
                select(QueryUsage.id)
                .where(QueryUsage.created_at < cutoff)
                .order_by(QueryUsage.created_at)
                .limit(_BATCH_SIZE)
            )
            result = await db.execute(id_stmt)
            ids = [row[0] for row in result.all()]

            if not ids:
                break

            del_stmt = (
                delete(QueryUsage)
                .where(QueryUsage.id.in_(ids))
            )
            del_result = await db.execute(del_stmt)
            await db.commit()

            batch_count = del_result.rowcount  # type: ignore[union-attr]
            total_deleted += batch_count

            logger.info(
                "retention.batch deleted=%d cumulative=%d",
                batch_count,
                total_deleted,
            )

            # If we got fewer than BATCH_SIZE, we're done
            if len(ids) < _BATCH_SIZE:
                break

        logger.info("retention.done total_deleted=%d", total_deleted)
        return total_deleted
