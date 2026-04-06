"""
Monthly cost summary service (Phase 4.3).

Validates year/month, aggregates usage via the repository, and returns
a structured DTO.  Cost estimation is token-based (price config optional).

🚫 Never exposes or stores raw query text.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.repos.query_cost_repo import QueryCostRepository

logger = logging.getLogger(__name__)

# Reasonable bounds
_MIN_YEAR = 2020
_MAX_YEAR = 2100


class QueryCostService:
    """
    Thin service layer over ``QueryCostRepository``.

    Responsibilities:
      * Validate year / month.
      * Delegate to repository for SQL aggregation.
      * Format the month label (``YYYY-MM``).
    """

    __slots__ = ("_repo",)

    def __init__(self, repo: QueryCostRepository | None = None) -> None:
        self._repo = repo or QueryCostRepository()

    # ── validation ────────────────────────────────────────────────────

    @staticmethod
    def validate_year_month(year: int, month: int) -> None:
        """Raise ``ValueError`` if year/month are out of range."""
        if not (_MIN_YEAR <= year <= _MAX_YEAR):
            raise ValueError(f"year must be between {_MIN_YEAR} and {_MAX_YEAR}")
        if not (1 <= month <= 12):
            raise ValueError("month must be between 1 and 12")

    # ── public API ────────────────────────────────────────────────────

    async def get_monthly_cost(
        self,
        db: AsyncSession,
        *,
        tenant_id: str,
        year: int,
        month: int,
    ) -> dict[str, Any]:
        """
        Monthly cost summary for a tenant.

        Returns::

            {
                "tenant_id": str,
                "month": "YYYY-MM",
                "total_queries": int,
                "total_tokens": int,
                "avg_latency_ms": float,
            }
        """
        self.validate_year_month(year, month)

        summary = await self._repo.get_monthly_cost_summary(
            db,
            tenant_id=tenant_id,
            year=year,
            month=month,
        )

        return {
            "tenant_id": tenant_id,
            "month": f"{year:04d}-{month:02d}",
            "total_queries": summary["total_queries"],
            "total_tokens": summary["total_tokens"],
            "avg_latency_ms": summary["avg_latency_ms"],
        }
