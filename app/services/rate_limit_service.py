"""
Rate-limit service (Phase 5.0).

Enforces tenant-level (and optionally user-level) rate limits
using a DB-backed token bucket.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.repos.rate_limit_repo import RateLimitRepository
from app.services.quota_policy_service import PolicyDTO

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RateLimitResult:
    """Result of a rate-limit check."""
    allowed: bool
    limit: int
    remaining: int
    reset_at: datetime

    def headers(self) -> dict[str, str]:
        """Return X-RateLimit-* headers for the HTTP response."""
        return {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(self.remaining),
            "X-RateLimit-Reset": str(int(self.reset_at.timestamp())),
        }


class RateLimitService:
    """
    Enforce rate limits via DB-backed token bucket.

    - Tenant bucket is always checked.
    - User bucket is checked only if enforce_user_rate_limit is True.
    """

    __slots__ = ("_repo",)

    def __init__(self, repo: RateLimitRepository | None = None) -> None:
        self._repo = repo or RateLimitRepository()

    async def check_rate_limit(
        self,
        db: AsyncSession,
        *,
        tenant_id: str,
        user_id: int | None,
        policy: PolicyDTO,
        now: datetime,
    ) -> RateLimitResult:
        """
        Enforce tenant (and optionally user) rate limit.

        Returns the most restrictive result.
        """
        if not settings.RATE_LIMIT_ENABLED:
            return RateLimitResult(
                allowed=True,
                limit=policy.per_minute + policy.burst,
                remaining=policy.per_minute + policy.burst,
                reset_at=now,
            )

        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        capacity = policy.per_minute + policy.burst
        window_sec = settings.RATE_LIMIT_WINDOW_SEC
        bucket_key = settings.RATE_LIMIT_BUCKET_KEY

        # 1. Tenant-level bucket
        tenant_result = await self._repo.consume_token(
            db,
            tenant_id=tenant_id,
            user_id=None,
            scope="tenant",
            bucket_key=bucket_key,
            window_sec=window_sec,
            capacity=capacity,
            now=now,
        )

        if not tenant_result.allowed:
            return RateLimitResult(
                allowed=False,
                limit=capacity,
                remaining=0,
                reset_at=tenant_result.reset_at,
            )

        # 2. User-level bucket (optional)
        if policy.enforce_user_rate_limit and user_id is not None:
            user_result = await self._repo.consume_token(
                db,
                tenant_id=tenant_id,
                user_id=user_id,
                scope="user",
                bucket_key=bucket_key,
                window_sec=window_sec,
                capacity=capacity,
                now=now,
            )
            if not user_result.allowed:
                return RateLimitResult(
                    allowed=False,
                    limit=capacity,
                    remaining=0,
                    reset_at=user_result.reset_at,
                )
            # Return the most restrictive remaining
            return RateLimitResult(
                allowed=True,
                limit=capacity,
                remaining=min(tenant_result.remaining, user_result.remaining),
                reset_at=min(tenant_result.reset_at, user_result.reset_at),
            )

        return RateLimitResult(
            allowed=True,
            limit=capacity,
            remaining=tenant_result.remaining,
            reset_at=tenant_result.reset_at,
        )
