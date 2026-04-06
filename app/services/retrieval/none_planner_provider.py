from __future__ import annotations


class NonePlannerProvider:
    async def plan(self, query_text: str) -> dict:
        return {}
