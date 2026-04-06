from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _collapse_ws(value: str) -> str:
    return " ".join((value or "").strip().split())


class QueryPlanFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_ids: list[int] = Field(default_factory=list, max_length=200)

    @field_validator("doc_ids", mode="before")
    @classmethod
    def _sanitize_doc_ids(cls, value):
        if not isinstance(value, list):
            return []
        out: list[int] = []
        seen: set[int] = set()
        for item in value:
            try:
                n = int(item)
            except Exception:
                continue
            if n <= 0 or n in seen:
                continue
            seen.add(n)
            out.append(n)
            if len(out) >= 200:
                break
        return out


class QueryPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    normalized_query: str
    subqueries: list[str] = Field(default_factory=list, max_length=3)
    filters: QueryPlanFilters = Field(default_factory=QueryPlanFilters)
    preferred_mode: Literal["vector", "bm25", "hybrid"] | None = None

    @field_validator("normalized_query")
    @classmethod
    def _sanitize_normalized_query(cls, value: str) -> str:
        text = _collapse_ws(value)[:1200]
        if not text:
            raise ValueError("normalized_query must be non-empty")
        return text

    @field_validator("subqueries", mode="before")
    @classmethod
    def _sanitize_subqueries(cls, value):
        if not isinstance(value, list):
            return []
        out: list[str] = []
        seen: set[str] = set()
        for item in value:
            text = _collapse_ws(str(item))[:120]
            if not text or text in seen:
                continue
            seen.add(text)
            out.append(text)
            if len(out) >= 3:
                break
        return out

    @classmethod
    def fallback(cls, query_text: str) -> "QueryPlan":
        text = _collapse_ws(query_text)[:1200] or "query"
        return cls(
            normalized_query=text,
            subqueries=[],
            filters=QueryPlanFilters(),
            preferred_mode=None,
        )
