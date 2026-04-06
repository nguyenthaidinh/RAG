from __future__ import annotations

import logging

from app.core.config import settings
from app.services.retrieval.plan_guard import finalize_plan, sanitize_text
from app.services.retrieval.planner_provider import PlannerProvider
from app.services.retrieval.query_plan import QueryPlan

logger = logging.getLogger(__name__)


class QueryPlanner:
    __slots__ = ("_provider", "_enabled", "_cache")

    def __init__(
        self,
        *,
        provider: PlannerProvider,
        enabled: bool,
        cache=None,
    ) -> None:
        self._provider = provider
        self._enabled = bool(enabled)
        self._cache = cache

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def build_plan(self, *, tenant_id: str, query_text: str) -> QueryPlan:
        try:
            max_chars = max(1, int(getattr(settings, "LLM_QUERY_PLANNER_MAX_QUERY_CHARS", 1200)))
            query_norm = sanitize_text(query_text, max_chars)
            if not self._enabled:
                return QueryPlan.fallback(query_norm)

            if self._cache is not None:
                cached = self._cache.get(tenant_id, query_norm)
                if cached is not None:
                    return cached

            raw = await self._provider.plan(query_norm)
            plan = finalize_plan(raw, query_norm)

            if self._cache is not None:
                self._cache.set(tenant_id, query_norm, plan)

            logger.info(
                "retrieval.query_planner planner_success=%s num_subqueries=%d num_doc_ids=%d fallback_reason=%s",
                bool(raw),
                len(plan.subqueries),
                len(plan.filters.doc_ids),
                "none" if raw else "empty_provider_response",
            )
            return plan
        except Exception:
            logger.warning(
                "retrieval.query_planner planner_success=false num_subqueries=0 num_doc_ids=0 fallback_reason=exception"
            )
            return QueryPlan.fallback(query_text)
