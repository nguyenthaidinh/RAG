"""
Token quota pre-check service (Phase 5.0).

Reads usage from TokenLedger totals to check if a tenant has exceeded
their daily/monthly token quota BEFORE executing expensive work.

This is a PRE-CHECK only — no double-charging. Billing remains Phase 4.1.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from app.core.config import settings
from app.services.quota_policy_service import PolicyDTO
from app.services.token_ledger import TokenLedgerService

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QuotaCheckResult:
    """Result of a token quota pre-check."""
    allowed: bool
    reason: str | None = None          # e.g. "daily_quota_exceeded"
    current_usage: int = 0
    limit: int | None = None


class TokenQuotaService:
    """
    Pre-check token quotas against TokenLedger totals.

    Controlled by TOKEN_QUOTA_PRECHECK_ENABLED config.
    When disabled, always returns allowed=True.
    """

    __slots__ = ("_ledger",)

    def __init__(self, ledger: TokenLedgerService) -> None:
        self._ledger = ledger

    async def check_quota(
        self,
        *,
        tenant_id: str,
        tokens_estimate: int,
        policy: PolicyDTO,
        now: datetime,
    ) -> QuotaCheckResult:
        """
        Check if adding *tokens_estimate* would exceed the tenant's
        daily or monthly quota.

        Returns allowed=True if no quota is configured or quotas are disabled.
        """
        if not settings.TOKEN_QUOTA_PRECHECK_ENABLED:
            return QuotaCheckResult(allowed=True)

        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        # Check daily quota
        if policy.token_daily is not None:
            day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            day_end = day_start + timedelta(days=1)
            daily_used = await self._ledger.sum_tokens(
                tenant_id=tenant_id,
                since=day_start,
                until=day_end,
            )
            if daily_used + tokens_estimate > policy.token_daily:
                return QuotaCheckResult(
                    allowed=False,
                    reason="daily_quota_exceeded",
                    current_usage=daily_used,
                    limit=policy.token_daily,
                )

        # Check monthly quota
        if policy.token_monthly is not None:
            month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            if now.month == 12:
                month_end = month_start.replace(year=now.year + 1, month=1)
            else:
                month_end = month_start.replace(month=now.month + 1)
            monthly_used = await self._ledger.sum_tokens(
                tenant_id=tenant_id,
                since=month_start,
                until=month_end,
            )
            if monthly_used + tokens_estimate > policy.token_monthly:
                return QuotaCheckResult(
                    allowed=False,
                    reason="monthly_quota_exceeded",
                    current_usage=monthly_used,
                    limit=policy.token_monthly,
                )

        return QuotaCheckResult(allowed=True)
