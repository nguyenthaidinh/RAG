"""
Plan repository (Phase 5.0).

CRUD for the plans table.
"""
from __future__ import annotations

import logging
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.plan import Plan

logger = logging.getLogger(__name__)

# Default "free" plan limits
DEFAULT_FREE_LIMITS = {
    "query_rate": {"per_minute": 120, "burst": 60},
    "token_quota": {"daily": None, "monthly": None},
    "max_users": None,
}


class PlanRepository:
    """Repository for plan management."""

    __slots__ = ()

    async def get_by_code(self, db: AsyncSession, code: str) -> Plan | None:
        """Return a plan by its unique code."""
        result = await db.execute(
            select(Plan).where(Plan.code == code)
        )
        return result.scalars().first()

    async def list_all(self, db: AsyncSession) -> list[Plan]:
        """Return all plans, ordered by created_at."""
        result = await db.execute(
            select(Plan).order_by(Plan.created_at)
        )
        return list(result.scalars().all())

    async def upsert_plan(
        self,
        db: AsyncSession,
        *,
        code: str,
        name: str,
        limits_json: dict,
        is_active: bool = True,
    ) -> Plan:
        """Create or update a plan by code."""
        existing = await self.get_by_code(db, code)
        if existing is not None:
            existing.name = name
            existing.limits_json = limits_json
            existing.is_active = is_active
            await db.flush()
            return existing

        plan = Plan(
            id=uuid.uuid4(),
            code=code,
            name=name,
            is_active=is_active,
            limits_json=limits_json,
        )
        db.add(plan)
        await db.flush()
        return plan
