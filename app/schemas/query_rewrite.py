"""
Query rewrite schemas (Phase 3A).

Defines the RetrievalPlan output and QueryMode classification.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class QueryMode(str, Enum):
    """Classification of user query intent for rewrite strategy."""

    DIRECT = "direct"                     # simple, clear, no rewrite needed
    OVERVIEW = "overview"                 # summary / broad question
    SPECIFIC = "specific"                 # detailed / pinpoint question
    COMPARISON = "comparison"             # comparing items
    FOLLOW_UP = "follow_up"              # references prior conversation
    AMBIGUOUS = "ambiguous"               # vague, needs clarification
    MULTI_HOP = "multi_hop"              # requires decomposition
    CONSTRAINT_HEAVY = "constraint_heavy" # deprecated: kept for compat, not emitted by classifier


class RewriteStrategy(str, Enum):
    """Gating decision for how aggressively to rewrite.

    Ordered from least to most invasive.
    """

    NO_REWRITE = "no_rewrite"                       # query is clear, pass through
    LIGHT_NORMALIZE = "light_normalize"             # conservative pass-through, no LLM rewrite
    CONTEXTUAL_REWRITE = "contextual_rewrite"       # use history to resolve references
    CONTROLLED_DECOMPOSITION = "controlled_decomposition"  # split into effective subqueries
    SAFE_FALLBACK = "safe_fallback"                 # insufficient confidence, keep original


@dataclass(frozen=True)
class RetrievalPlan:
    """
    Output of QueryRewriteService.

    Invariants:
      - original_query is always non-empty
      - rewritten_query is optional (None = use original only)
      - subqueries max 2 items
      - confidence 0.0 .. 1.0
    """

    original_query: str
    query_mode: QueryMode = QueryMode.DIRECT

    # Optional enhanced queries
    rewritten_query: str | None = None
    step_back_query: str | None = None
    subqueries: tuple[str, ...] = ()

    # Strategy
    rewrite_strategy: str = "no_rewrite"  # RewriteStrategy value
    strategy_flags: dict = field(default_factory=dict)
    confidence: float = 1.0
    rewrite_reason: str = ""
    used_history: bool = False

    # Telemetry
    latency_ms: int = 0
    fallback_used: bool = False

    def effective_queries(self) -> list[str]:
        """
        Build the list of queries to run retrieval on.

        Order: original → rewritten → step_back → subqueries.
        Full dedupe (normalized whitespace comparison).
        Empty/whitespace-only entries are rejected.
        """
        queries: list[str] = []
        seen: set[str] = set()

        def _add(q: str | None) -> None:
            if not q:
                return
            q = q.strip()
            if not q:
                return
            norm = " ".join(q.lower().split())
            if norm in seen:
                return
            seen.add(norm)
            queries.append(q)

        # original always first
        _add(self.original_query)

        if not self.fallback_used:
            _add(self.rewritten_query)
            _add(self.step_back_query)
            for sq in self.subqueries:
                _add(sq)

        return queries

    def telemetry_dict(self) -> dict:
        """Safe telemetry payload — no raw text."""
        return {
            "query_mode": self.query_mode.value,
            "rewrite_strategy": self.rewrite_strategy,
            "rewrite_used": self.rewritten_query is not None,
            "step_back_used": self.step_back_query is not None,
            "rewrite_confidence": round(self.confidence, 3),
            "subquery_count": len(self.subqueries),
            "fallback_used": self.fallback_used,
            "used_history": self.used_history,
            "latency_ms": self.latency_ms,
        }

    @classmethod
    def passthrough(cls, query: str) -> "RetrievalPlan":
        """No-op plan that just passes the original query through."""
        return cls(
            original_query=query,
            query_mode=QueryMode.DIRECT,
            confidence=1.0,
            fallback_used=True,
            rewrite_reason="passthrough",
        )
