from __future__ import annotations

from typing import Any

from app.core.config import settings
from app.services.retrieval.query_plan import QueryPlan, QueryPlanFilters


def sanitize_text(s: str, max_chars: int) -> str:
    text = " ".join((s or "").strip().split())
    if max_chars <= 0:
        return ""
    return text[:max_chars]


def sanitize_subqueries(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    max_subqueries = max(1, int(getattr(settings, "LLM_QUERY_PLANNER_MAX_SUBQUERIES", 3)))
    max_term_chars = max(1, int(getattr(settings, "LLM_QUERY_PLANNER_MAX_TERM_CHARS", 120)))
    for item in items or []:
        cleaned = sanitize_text(str(item), max_term_chars)
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        out.append(cleaned)
        if len(out) >= max_subqueries:
            break
    return out


def sanitize_doc_ids(items: list[Any]) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for item in items or []:
        try:
            value = int(item)
        except Exception:
            continue
        if value <= 0 or value in seen:
            continue
        seen.add(value)
        out.append(value)
        if len(out) >= 200:
            break
    return out


def finalize_plan(raw: dict, original_query: str) -> QueryPlan:
    if not isinstance(raw, dict):
        return QueryPlan.fallback(original_query)

    try:
        parsed = QueryPlan.model_validate(raw)
    except Exception:
        return QueryPlan.fallback(original_query)

    max_query_chars = max(1, int(getattr(settings, "LLM_QUERY_PLANNER_MAX_QUERY_CHARS", 1200)))
    normalized_query = sanitize_text(parsed.normalized_query, max_query_chars)
    if not normalized_query:
        return QueryPlan.fallback(original_query)

    subqueries = [s for s in sanitize_subqueries(parsed.subqueries) if s != normalized_query]
    doc_ids = sanitize_doc_ids(parsed.filters.doc_ids)

    return QueryPlan(
        normalized_query=normalized_query,
        subqueries=subqueries,
        filters=QueryPlanFilters(doc_ids=doc_ids),
        preferred_mode=parsed.preferred_mode,
    )
