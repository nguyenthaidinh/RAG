"""
Retrieval Execution Context — internal contract for retrieval pipeline state.

Captures all intermediate decisions and state produced during a single
retrieval execution in QueryService.query().  This is an INTERNAL contract
— not exposed in any public API or response schema.

Design rules:
  - Frozen dataclass: immutable after construction
  - No business logic — pure state container
  - No DB access, no I/O
  - All fields have safe defaults (None / empty / False)
  - telemetry_dict() for structured logging (no raw text)
  - Compatible with existing RetrievalPlan, MetadataPreference,
    RepresentationPreference types
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from app.schemas.query_rewrite import RetrievalPlan
    from app.schemas.retrieval_metadata import MetadataPreference
    from app.schemas.retrieval_representation import RepresentationPreference
    from app.services.retrieval.query_plan import QueryPlan


@dataclass(frozen=True)
class RetrievalExecutionContext:
    """
    Snapshot of all retrieval decisions for a single query execution.

    Constructed incrementally by QueryService — once fully built,
    it is frozen and used for execution, telemetry, and debugging.

    Fields are grouped by pipeline phase:
      1. Input normalization
      2. Query rewrite / plan
      3. Intent analysis
      4. Execution parameters

    All fields have safe defaults so partial construction is valid
    (e.g. when rewrite is disabled or planner fails).
    """

    # ── 1. Normalized inputs ──────────────────────────────────────────
    original_query: str = ""
    effective_mode: Literal["hybrid", "vector", "bm25"] = "hybrid"
    include_debug: bool = False

    # ── 2. Query rewrite / plan ───────────────────────────────────────
    rewrite_plan: RetrievalPlan | None = None
    rewrite_usable: bool = False
    query_plan: QueryPlan | None = None
    effective_queries: tuple[str, ...] = ()
    candidate_doc_ids: frozenset[int] = field(default_factory=frozenset)
    history_provided: bool = False

    # ── 3. Intent analysis (Phase 3B / 3D) ────────────────────────────
    metadata_preference: MetadataPreference | None = None
    representation_preference: RepresentationPreference | None = None

    # ── Telemetry ─────────────────────────────────────────────────────

    def telemetry_dict(self) -> dict:
        """Safe telemetry payload — no raw query text or PII."""
        return {
            "effective_mode": self.effective_mode,
            "include_debug": self.include_debug,
            "rewrite_usable": self.rewrite_usable,
            "rewrite_plan_present": self.rewrite_plan is not None,
            "query_plan_present": self.query_plan is not None,
            "effective_query_count": len(self.effective_queries),
            "candidate_doc_count": len(self.candidate_doc_ids),
            "history_provided": self.history_provided,
            "metadata_pref_present": self.metadata_preference is not None,
            "representation_pref_present": (
                self.representation_preference is not None
            ),
        }

    @classmethod
    def empty(cls) -> RetrievalExecutionContext:
        """Minimal context for error/abort paths."""
        return cls()
