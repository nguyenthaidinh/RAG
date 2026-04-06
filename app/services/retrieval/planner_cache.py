from __future__ import annotations

import hashlib
import time
from collections import OrderedDict

from app.services.retrieval.query_plan import QueryPlan


class PlannerCache:
    __slots__ = ("_ttl_s", "_max_entries", "_store")

    def __init__(self, ttl_s: int, max_entries: int = 1000) -> None:
        self._ttl_s = max(1, int(ttl_s))
        self._max_entries = max(1, int(max_entries))
        self._store: OrderedDict[tuple[str, str], tuple[float, QueryPlan]] = OrderedDict()

    @staticmethod
    def _key(tenant_id: str, query_text: str) -> tuple[str, str]:
        digest = hashlib.sha256((query_text or "").encode("utf-8")).hexdigest()
        return (tenant_id, digest)

    def get(self, tenant_id: str, query_text: str) -> QueryPlan | None:
        key = self._key(tenant_id, query_text)
        item = self._store.get(key)
        if item is None:
            return None
        expires_at, plan = item
        now = time.time()
        if expires_at <= now:
            self._store.pop(key, None)
            return None
        self._store.move_to_end(key)
        return plan

    def set(self, tenant_id: str, query_text: str, plan: QueryPlan) -> None:
        key = self._key(tenant_id, query_text)
        expires_at = time.time() + self._ttl_s
        self._store[key] = (expires_at, plan)
        self._store.move_to_end(key)
        while len(self._store) > self._max_entries:
            self._store.popitem(last=False)
