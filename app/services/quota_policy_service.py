"""
Quota policy service (Phase 5.0).

Resolves effective policy for a tenant by deep-merging the plan's
limits_json with the tenant's quota_overrides_json.
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.repos.plan_repo import PlanRepository, DEFAULT_FREE_LIMITS
from app.repos.tenant_setting_repo import TenantSettingRepository

logger = logging.getLogger(__name__)


@dataclass
class PolicyDTO:
    """Resolved effective quota/rate-limit policy for a tenant."""

    per_minute: int
    burst: int
    token_daily: int | None
    token_monthly: int | None
    max_users: int | None
    enforce_user_rate_limit: bool


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep-merge *override* into *base* (non-mutating)."""
    result = copy.deepcopy(base)
    for key, val in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(val, dict)
        ):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


class QuotaPolicyService:
    """
    Resolve effective policy for a tenant.

    Priority: tenant_settings.quota_overrides_json > plan.limits_json > defaults.
    """

    __slots__ = ("_plan_repo", "_setting_repo")

    def __init__(
        self,
        plan_repo: PlanRepository | None = None,
        setting_repo: TenantSettingRepository | None = None,
    ) -> None:
        self._plan_repo = plan_repo or PlanRepository()
        self._setting_repo = setting_repo or TenantSettingRepository()

    async def get_effective_policy(
        self,
        db: AsyncSession,
        tenant_id: str,
    ) -> PolicyDTO:
        """
        Resolve effective policy for *tenant_id*.

        Falls back to default free plan if plan/settings are missing.
        """
        # 1. Get tenant settings (may be None)
        tenant_setting = await self._setting_repo.get(db, tenant_id)

        plan_code = settings.DEFAULT_PLAN_CODE
        overrides: dict[str, Any] = {}
        enforce_user_rl = False

        if tenant_setting is not None:
            plan_code = tenant_setting.plan_code or plan_code
            overrides = tenant_setting.quota_overrides_json or {}
            enforce_user_rl = tenant_setting.enforce_user_rate_limit

        # 2. Get plan limits
        plan = await self._plan_repo.get_by_code(db, plan_code)
        if plan is not None and plan.is_active:
            base_limits = plan.limits_json
        else:
            base_limits = DEFAULT_FREE_LIMITS

        # 3. Deep merge overrides on top of plan limits
        merged = _deep_merge(base_limits, overrides)

        # 4. Build DTO
        rate = merged.get("query_rate", {})
        quota = merged.get("token_quota", {})

        return PolicyDTO(
            per_minute=rate.get("per_minute", 120),
            burst=rate.get("burst", 60),
            token_daily=quota.get("daily"),
            token_monthly=quota.get("monthly"),
            max_users=merged.get("max_users"),
            enforce_user_rate_limit=enforce_user_rl,
        )
