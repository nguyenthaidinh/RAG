"""
Quota enforcement orchestrator (Phase 5.0).

Single entry point for rate-limit + token-quota enforcement,
called BEFORE executing expensive query work.

Key enterprise requirement:
  Idempotent retries MUST NEVER be blocked by rate-limit/quota.

Phase 6.0 hooks: emits audit events on denial (RATE_LIMIT_HIT,
TOKEN_QUOTA_EXCEEDED).  Failures in audit do NOT break enforcement.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.repos.query_usage_repo import QueryUsageRepository
from app.services.quota_policy_service import PolicyDTO, QuotaPolicyService
from app.services.rate_limit_service import RateLimitResult, RateLimitService
from app.services.token_quota_service import TokenQuotaService

logger = logging.getLogger(__name__)


class QuotaEnforcementService:
    """
    Orchestrate rate-limit + token-quota enforcement before query execution.

    Process:
      1. Idempotent retry bypass
      2. Resolve effective policy
      3. Rate limit check
      4. Token quota pre-check
    """

    __slots__ = (
        "_audit_svc",
        "_policy_svc",
        "_rate_limit_svc",
        "_token_quota_svc",
        "_usage_repo",
    )

    def __init__(
        self,
        *,
        policy_svc: QuotaPolicyService,
        rate_limit_svc: RateLimitService,
        token_quota_svc: TokenQuotaService,
        usage_repo: QueryUsageRepository,
        audit_svc=None,
    ) -> None:
        self._policy_svc = policy_svc
        self._rate_limit_svc = rate_limit_svc
        self._token_quota_svc = token_quota_svc
        self._usage_repo = usage_repo
        self._audit_svc = audit_svc  # injected or lazy-loaded

    def _get_audit_svc(self):
        """Lazy-load the audit service to avoid circular imports."""
        if self._audit_svc is None:
            from app.services.audit_service import get_audit_service
            self._audit_svc = get_audit_service()
        return self._audit_svc

    async def enforce_or_raise(
        self,
        db: AsyncSession,
        *,
        tenant_id: str,
        user_id: int | None,
        idempotency_key: str,
        tokens_estimate: int,
        now: datetime | None = None,
        request_id: str | None = None,
    ) -> RateLimitResult:
        """
        Enforce rate limit and token quota.

        Returns the rate-limit result (for headers) on success.
        Raises HTTPException(429) on denial.

        Idempotent retries (existing idempotency_key) BYPASS all checks.
        """
        if not settings.QUOTA_ENABLED:
            return RateLimitResult(
                allowed=True,
                limit=0,
                remaining=0,
                reset_at=now or datetime.now(timezone.utc),
            )

        if now is None:
            now = datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        # ── 1. Idempotent retry bypass ─────────────────────────────
        existing = await self._usage_repo.get_by_idempotency(
            tenant_id=tenant_id,
            idempotency_key=idempotency_key,
        )
        if existing is not None:
            logger.info(
                "quota.bypass_idempotent tenant_id=%s idempotency_key=%s",
                tenant_id, idempotency_key,
            )
            return RateLimitResult(
                allowed=True,
                limit=0,
                remaining=0,
                reset_at=now,
            )

        # ── 2. Resolve policy ─────────────────────────────────────
        policy = await self._policy_svc.get_effective_policy(db, tenant_id)

        # ── 3. Rate limit check ──────────────────────────────────
        rl_result = await self._rate_limit_svc.check_rate_limit(
            db,
            tenant_id=tenant_id,
            user_id=user_id,
            policy=policy,
            now=now,
        )
        if not rl_result.allowed:
            # ── Phase 6.0: emit RATE_LIMIT_HIT audit (failure-isolated) ─
            try:
                await self._get_audit_svc().log_rate_limit_hit(
                    db,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    request_id=request_id,
                    limit=rl_result.limit,
                    remaining=rl_result.remaining,
                    reset_at=rl_result.reset_at.isoformat(),
                    scope="tenant",
                    now=now,
                )
            except Exception:
                logger.warning("audit.rate_limit_hit_failed tenant_id=%s", tenant_id, exc_info=True)

            # ── Phase 7.0: metrics hook (best-effort) ─
            try:
                from app.core.metrics import observe_rate_limit_hit
                observe_rate_limit_hit(tenant_id=tenant_id, scope="tenant")
            except Exception:
                pass

            raise HTTPException(
                status_code=429,
                detail={
                    "error": "rate_limit_exceeded",
                    "message": "Too many requests. Please retry after the reset window.",
                    "reset_at": rl_result.reset_at.isoformat(),
                },
                headers=rl_result.headers(),
            )

        # ── 4. Token quota pre-check ─────────────────────────────
        quota_result = await self._token_quota_svc.check_quota(
            tenant_id=tenant_id,
            tokens_estimate=tokens_estimate,
            policy=policy,
            now=now,
        )
        if not quota_result.allowed:
            # ── Phase 6.0: emit TOKEN_QUOTA_EXCEEDED audit (failure-isolated) ─
            try:
                await self._get_audit_svc().log_quota_exceeded(
                    db,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    request_id=request_id,
                    reason_code=quota_result.reason or "quota_exceeded",
                    tokens_total=quota_result.current_usage,
                    limit=quota_result.limit,
                    now=now,
                )
            except Exception:
                logger.warning("audit.quota_exceeded_failed tenant_id=%s", tenant_id, exc_info=True)

            # ── Phase 7.0: metrics hook (best-effort) ─
            try:
                from app.core.metrics import observe_token_quota_exceeded
                observe_token_quota_exceeded(tenant_id=tenant_id)
            except Exception:
                pass

            raise HTTPException(
                status_code=429,
                detail={
                    "error": "quota_exceeded",
                    "reason": quota_result.reason,
                    "message": f"Token quota exceeded: {quota_result.reason}",
                    "current_usage": quota_result.current_usage,
                    "limit": quota_result.limit,
                },
                headers=rl_result.headers(),
            )

        return rl_result
