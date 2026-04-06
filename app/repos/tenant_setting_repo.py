"""
Tenant settings repository (Phase 5.0).

CRUD for the tenant_settings table.
"""
from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.tenant_setting import TenantSetting

logger = logging.getLogger(__name__)


class TenantSettingRepository:
    """Repository for per-tenant plan/quota settings."""

    __slots__ = ()

    async def get(self, db: AsyncSession, tenant_id: str) -> TenantSetting | None:
        """Return tenant settings, or None if not yet configured."""
        result = await db.execute(
            select(TenantSetting).where(TenantSetting.tenant_id == tenant_id)
        )
        return result.scalars().first()

    async def upsert(
        self,
        db: AsyncSession,
        *,
        tenant_id: str,
        plan_code: str | None = None,
        quota_overrides_json: dict | None = None,
        enforce_user_rate_limit: bool | None = None,
    ) -> TenantSetting:
        """Create or update tenant settings."""
        existing = await self.get(db, tenant_id)
        if existing is not None:
            if plan_code is not None:
                existing.plan_code = plan_code
            if quota_overrides_json is not None:
                existing.quota_overrides_json = quota_overrides_json
            if enforce_user_rate_limit is not None:
                existing.enforce_user_rate_limit = enforce_user_rate_limit
            await db.flush()
            return existing

        setting = TenantSetting(
            tenant_id=tenant_id,
            plan_code=plan_code or "free",
            quota_overrides_json=quota_overrides_json,
            enforce_user_rate_limit=enforce_user_rate_limit or False,
        )
        db.add(setting)
        await db.flush()
        return setting
