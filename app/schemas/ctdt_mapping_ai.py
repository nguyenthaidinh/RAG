"""
CTDT Mapping AI — Request/Response schemas for R6.13C1.

POST /api/v1/ctdt/update-cycles/mapping-draft/ai-build

Validates:
  - approved_objective_snapshot: is_completed, M-codes, unique codes, non-empty text.
  - approved_outcome_snapshot:  is_completed, C-codes, unique codes, non-empty text,
    group ∈ {knowledge, skills, autonomy_responsibility}.
  - top_k bounds.
  - Technical size bounds on snapshot items to prevent oversized LLM payloads.

Guards:
  - Does NOT modify snapshot content.
  - Does NOT add/remove M/C codes.
  - Does NOT change CĐR groups.
"""
from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field, model_validator

# ── Allowed CĐR groups ──────────────────────────────────────────────

ALLOWED_OUTCOME_GROUPS = frozenset([
    "knowledge",
    "skills",
    "autonomy_responsibility",
])

# ── Code patterns ────────────────────────────────────────────────────

_M_CODE_RE = re.compile(r"^M\d+$", re.IGNORECASE)
_C_CODE_RE = re.compile(r"^C\d+$", re.IGNORECASE)


# ── Snapshot sub-schemas ─────────────────────────────────────────────


class ObjectiveSnapshotItem(BaseModel):
    code: str = Field(min_length=1, max_length=16)
    text: str = Field(min_length=1, max_length=4000)


class ApprovedObjectiveSnapshot(BaseModel):
    is_completed: bool
    general_objective: str | None = Field(default=None, max_length=8000)
    specific_objectives: list[ObjectiveSnapshotItem] = Field(
        min_length=1,
        max_length=30,
    )


class OutcomeSnapshotItem(BaseModel):
    code: str = Field(min_length=1, max_length=16)
    group: str = Field(min_length=1, max_length=64)
    text: str = Field(min_length=1, max_length=4000)


class ApprovedOutcomeSnapshot(BaseModel):
    is_completed: bool
    outcomes: list[OutcomeSnapshotItem] = Field(
        min_length=1,
        max_length=50,
    )


# ── Request ──────────────────────────────────────────────────────────


class MappingAIBuildRequest(BaseModel):
    """Request body for POST /update-cycles/mapping-draft/ai-build."""

    update_cycle_id: str = Field(min_length=1, max_length=64)
    program_id: str | None = Field(default=None, max_length=64)
    program_code: str | None = Field(default=None, max_length=64)
    program_name: str | None = Field(default=None, max_length=256)
    analysis_mode: str = Field(default="design")

    approved_objective_snapshot: ApprovedObjectiveSnapshot
    approved_outcome_snapshot: ApprovedOutcomeSnapshot

    top_k: int = Field(default=8, ge=1, le=20)
    user_instruction: str | None = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def validate_snapshots(self) -> "MappingAIBuildRequest":
        errors: list[str] = []

        # ── analysis_mode ────────────────────────────────────────
        if self.analysis_mode != "design":
            errors.append(
                "analysis_mode phải là 'design'."
            )

        # ── Objective snapshot ───────────────────────────────────
        obj = self.approved_objective_snapshot
        if not obj.is_completed:
            errors.append(
                "approved_objective_snapshot.is_completed phải là true."
            )
        if not obj.specific_objectives:
            errors.append(
                "approved_objective_snapshot.specific_objectives không được rỗng."
            )
        else:
            seen_obj_codes: set[str] = set()
            for idx, item in enumerate(obj.specific_objectives):
                code_upper = (item.code or "").strip().upper()
                if not code_upper:
                    errors.append(
                        f"specific_objectives[{idx}].code không được rỗng."
                    )
                elif not _M_CODE_RE.match(code_upper):
                    errors.append(
                        f"specific_objectives[{idx}].code='{item.code}' "
                        f"không hợp lệ; phải theo pattern M + số (vd: M1, M2)."
                    )
                if code_upper in seen_obj_codes:
                    errors.append(
                        f"specific_objectives[{idx}].code='{item.code}' bị trùng."
                    )
                seen_obj_codes.add(code_upper)
                if not (item.text or "").strip():
                    errors.append(
                        f"specific_objectives[{idx}].text không được rỗng."
                    )

        # ── Outcome snapshot ─────────────────────────────────────
        out = self.approved_outcome_snapshot
        if not out.is_completed:
            errors.append(
                "approved_outcome_snapshot.is_completed phải là true."
            )
        if not out.outcomes:
            errors.append(
                "approved_outcome_snapshot.outcomes không được rỗng."
            )
        else:
            seen_out_codes: set[str] = set()
            for idx, item in enumerate(out.outcomes):
                code_upper = (item.code or "").strip().upper()
                if not code_upper:
                    errors.append(
                        f"outcomes[{idx}].code không được rỗng."
                    )
                elif not _C_CODE_RE.match(code_upper):
                    errors.append(
                        f"outcomes[{idx}].code='{item.code}' "
                        f"không hợp lệ; phải theo pattern C + số (vd: C1, C2)."
                    )
                if code_upper in seen_out_codes:
                    errors.append(
                        f"outcomes[{idx}].code='{item.code}' bị trùng."
                    )
                seen_out_codes.add(code_upper)
                if not (item.text or "").strip():
                    errors.append(
                        f"outcomes[{idx}].text không được rỗng."
                    )
                if item.group not in ALLOWED_OUTCOME_GROUPS:
                    errors.append(
                        f"outcomes[{idx}].group='{item.group}' không hợp lệ; "
                        f"phải thuộc: {', '.join(sorted(ALLOWED_OUTCOME_GROUPS))}."
                    )

        if errors:
            raise ValueError(
                "Snapshot validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

        return self


# ── Response sub-schemas ─────────────────────────────────────────────


class MappingAIEntry(BaseModel):
    """A single AI-proposed mapping link."""
    objective_code: str
    outcome_code: str
    reason: str
    confidence: str = "medium"


class MappingAICoverage(BaseModel):
    """Coverage summary of the AI mapping result."""
    objective_codes: list[str]
    outcome_codes: list[str]
    mapped_objective_codes: list[str]
    mapped_outcome_codes: list[str]
    unmapped_objective_codes: list[str]
    unmapped_outcome_codes: list[str]
    mapping_count: int


class MappingAISourceSummary(BaseModel):
    """RAG retrieval source metadata."""
    task_type: str = "matrix_mapping"
    retrieved_chunk_count: int = 0
    used_chunk_count: int = 0
    document_roles: list[str] = Field(default_factory=list)
    latency_ms: int = 0


# ── Response ─────────────────────────────────────────────────────────


class MappingAIBuildResponse(BaseModel):
    """Response from POST /update-cycles/mapping-draft/ai-build."""

    status: str  # "generated" | "failed"
    source: str = "laravel_approved_snapshot"
    update_cycle_id: str
    program_id: str | None = None
    program_code: str | None = None
    program_name: str | None = None
    analysis_mode: str = "design"

    mappings: list[MappingAIEntry]
    coverage: MappingAICoverage
    quality_level: str  # "good" | "warning" | "failed"
    quality_messages: list[str] = Field(default_factory=list)

    retrieval_used: bool = False
    source_summary: MappingAISourceSummary = Field(
        default_factory=MappingAISourceSummary
    )
    warnings: list[str] = Field(default_factory=list)
