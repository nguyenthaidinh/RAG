from __future__ import annotations

from typing import Protocol


class PlannerProvider(Protocol):
    async def plan(self, query_text: str) -> dict:
        ...
