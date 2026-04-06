"""
Rate-limit bucket repository (Phase 5.0).

Atomic token bucket with SELECT FOR UPDATE for multi-process safety.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.rate_limit_bucket import RateLimitBucket

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConsumeResult:
    """Result of a token bucket consume attempt."""
    allowed: bool
    remaining: int
    reset_at: datetime


class RateLimitRepository:
    """
    DB-backed token bucket rate limiter.

    Uses SELECT FOR UPDATE for multi-process safety.
    """

    __slots__ = ()

    async def consume_token(
        self,
        db: AsyncSession,
        *,
        tenant_id: str,
        user_id: int | None,
        scope: str,
        bucket_key: str,
        window_sec: int,
        capacity: int,
        now: datetime,
    ) -> ConsumeResult:
        """
        Attempt to consume one token from the bucket.

        Rules:
        - If bucket missing OR reset_at <= now: reset tokens=capacity, reset_at=now+window
        - If tokens <= 0: denied
        - Else tokens -= 1: allowed
        """
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        # Try to find existing bucket with lock
        stmt = (
            select(RateLimitBucket)
            .where(
                RateLimitBucket.tenant_id == tenant_id,
                RateLimitBucket.user_id == user_id if user_id is not None
                else RateLimitBucket.user_id.is_(None),
                RateLimitBucket.scope == scope,
                RateLimitBucket.bucket_key == bucket_key,
                RateLimitBucket.window_sec == window_sec,
            )
            .with_for_update()
        )
        result = await db.execute(stmt)
        bucket = result.scalars().first()

        if bucket is None or bucket.reset_at <= now:
            # Reset or create bucket
            new_reset = now + timedelta(seconds=window_sec)
            new_tokens = capacity - 1  # consume one token immediately

            if bucket is None:
                bucket = RateLimitBucket(
                    id=uuid.uuid4(),
                    tenant_id=tenant_id,
                    user_id=user_id,
                    scope=scope,
                    window_sec=window_sec,
                    bucket_key=bucket_key,
                    tokens=new_tokens,
                    reset_at=new_reset,
                )
                db.add(bucket)
            else:
                bucket.tokens = new_tokens
                bucket.reset_at = new_reset

            await db.flush()
            return ConsumeResult(
                allowed=True,
                remaining=new_tokens,
                reset_at=new_reset,
            )

        # Bucket exists and is within window
        if bucket.tokens <= 0:
            return ConsumeResult(
                allowed=False,
                remaining=0,
                reset_at=bucket.reset_at,
            )

        bucket.tokens -= 1
        await db.flush()
        return ConsumeResult(
            allowed=True,
            remaining=bucket.tokens,
            reset_at=bucket.reset_at,
        )
