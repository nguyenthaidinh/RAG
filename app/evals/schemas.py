"""
Phase 1.2A+B — Eval case & result schemas.

Pydantic models for:
  - EvalCase: a single golden-set entry
  - CaseResult: per-case output with all metrics
  - RunSummary: aggregate report payload

Design rules:
  - tenant_id is **required** — enforced at validation time.
  - Retrieval metrics and answer metrics live in separate sections.
  - 1.2B judge/pairwise fields are optional with safe defaults.
  - All fields have safe defaults so partial results never crash reporters.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


# ── Golden-set input ──────────────────────────────────────────────────────────


class EvalCase(BaseModel):
    """A single evaluation case from golden_set.jsonl.

    Every field has a safe default so the parser can report which fields
    are missing rather than hard-crashing on load.
    """

    case_id: str
    tenant_id: str
    module: str | None = None

    endpoint_mode: Literal["assistant", "query"] = "assistant"

    question: str
    history: list[dict[str, str]] = Field(default_factory=list)

    expected_answer_type: Literal[
        "overview",
        "specific",
        "follow_up",
        "abstain",
        "ambiguous",
        "citation_required",
        # Backward compat with existing golden set
        "general",
        "compare",
        "no_answer",
    ] = "specific"

    expected_source_document_ids: list[int] = Field(default_factory=list)
    expected_keywords: list[str] = Field(default_factory=list)
    forbidden_keywords: list[str] = Field(default_factory=list)

    criticality: Literal["low", "medium", "high"] = "medium"
    slice_tags: list[str] = Field(default_factory=list)
    notes: str | None = None

    @field_validator("tenant_id")
    @classmethod
    def tenant_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("tenant_id must not be empty")
        return v.strip()

    @field_validator("case_id")
    @classmethod
    def case_id_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("case_id must not be empty")
        return v.strip()


# ── Per-case result ───────────────────────────────────────────────────────────


class RetrievalMetrics(BaseModel):
    """Retrieval-side metrics for a single case."""

    retrieved_document_ids: list[int] = Field(default_factory=list)
    hit_at_1: bool = False
    hit_at_3: bool = False
    hit_at_5: bool = False
    recall_at_5: float = 0.0
    mrr: float = 0.0


class AnswerMetrics(BaseModel):
    """Answer-side metrics for a single case."""

    has_answer: bool = False
    has_citations: bool = False
    citation_doc_ids_valid: bool = False
    citation_same_tenant: bool = False
    keyword_coverage: float = 0.0
    keywords_found: list[str] = Field(default_factory=list)
    keywords_missing: list[str] = Field(default_factory=list)
    forbidden_keyword_violation: bool = False
    forbidden_keywords_found: list[str] = Field(default_factory=list)
    abstention_detected: bool = False
    abstention_behavior_correct: bool = False


class CaseResult(BaseModel):
    """Full per-case result including raw outputs and computed metrics."""

    case_id: str
    tenant_id: str
    mode: str  # "assistant" | "query"
    question: str
    expected_answer_type: str
    expected_source_document_ids: list[int] = Field(default_factory=list)
    slice_tags: list[str] = Field(default_factory=list)
    criticality: str = "medium"

    # Raw outputs
    answer_text: str | None = None
    citation_document_ids: list[int] = Field(default_factory=list)
    retrieved_document_ids: list[int] = Field(default_factory=list)

    # Computed metrics
    retrieval: RetrievalMetrics = Field(default_factory=RetrievalMetrics)
    answer: AnswerMetrics = Field(default_factory=AnswerMetrics)

    # Execution metadata
    latency_ms: int = 0
    error: str | None = None
    failure_reasons: list[str] = Field(default_factory=list)
    skipped: bool = False
    skip_reason: str | None = None

    # 1.2B: Judge scores (populated only when --with-judges is used)
    judge_scores: dict[str, Any] = Field(default_factory=dict)
    # 1.2B: Pairwise result (populated only in pairwise mode)
    pairwise_result: dict[str, Any] | None = None


# ── Aggregate report ──────────────────────────────────────────────────────────


class AggregateMetrics(BaseModel):
    """Aggregated metrics for a group of cases."""

    count: int = 0

    # Retrieval
    retrieval_hit_at_1: float = 0.0
    retrieval_hit_at_3: float = 0.0
    retrieval_hit_at_5: float = 0.0
    retrieval_recall_at_5: float = 0.0
    retrieval_mrr: float = 0.0

    # Answer
    has_answer_rate: float = 0.0
    has_citations_rate: float = 0.0
    citation_doc_ids_valid_rate: float = 0.0
    citation_same_tenant_rate: float = 0.0
    keyword_coverage_avg: float = 0.0
    forbidden_keyword_violation_rate: float = 0.0
    abstention_behavior_accuracy: float = 0.0

    # Stability
    avg_latency_ms: float = 0.0
    error_count: int = 0

    # 1.2B: Judge aggregates (0 if judges not used)
    avg_faithfulness: float = 0.0
    avg_answer_relevance: float = 0.0
    judge_failures: int = 0


class RunSummary(BaseModel):
    """Top-level eval run summary (written to summary.json)."""

    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    git_commit: str | None = None
    mode: str  # "assistant" | "query" | "mixed"
    tenant: str | None = None

    evaluated_cases: int = 0
    skipped_cases: int = 0

    overall: AggregateMetrics = Field(default_factory=AggregateMetrics)
    by_tenant: dict[str, AggregateMetrics] = Field(default_factory=dict)
    by_slice: dict[str, AggregateMetrics] = Field(default_factory=dict)
    by_expected_answer_type: dict[str, AggregateMetrics] = Field(default_factory=dict)

    # 1.2B extensions
    judge_model: str | None = None
    judged_cases: int = 0
    pairwise: dict[str, Any] | None = None
