"""
Phase 1.2A — Dataset loader & validator.

Reads golden_set.jsonl, validates each line against EvalCase schema,
reports per-line errors, and supports rich filtering.

Design rules:
  - Never silently skip a bad line — always collect the error.
  - tenant_id is required; lines without it are always rejected.
  - Pydantic validation gives us type safety + clear error messages.
  - Filtering is composable: tenant, mode, case_id, slice, limit.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from app.evals.schemas import EvalCase

logger = logging.getLogger(__name__)


@dataclass
class DatasetLoadResult:
    """Result from loading a JSONL dataset."""

    cases: list[EvalCase] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)
    total_lines: int = 0
    valid_lines: int = 0
    skipped_by_filter: int = 0


def load_dataset(
    path: str | Path,
    *,
    tenant_id: str | None = None,
    endpoint_mode: str | None = None,
    case_id: str | None = None,
    slice_tag: str | None = None,
    limit: int | None = None,
) -> DatasetLoadResult:
    """Load and validate a golden-set JSONL file.

    Args:
        path: Path to JSONL file.
        tenant_id: If set, only return cases matching this tenant.
        endpoint_mode: If set, only return cases matching this mode
                       ("assistant" or "query").
        case_id: If set, only return the single case with this ID.
        slice_tag: If set, only return cases whose slice_tags contain
                   this tag.
        limit: Max number of cases to return (after filtering).

    Returns:
        DatasetLoadResult with validated cases and any parse/validation
        errors encountered.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    result = DatasetLoadResult()

    with open(path, "r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            result.total_lines += 1

            # ── Parse JSON ──
            try:
                data = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                result.errors.append({
                    "line": line_num,
                    "error": f"JSON parse error: {exc}",
                    "raw": raw_line[:200],
                })
                continue

            # ── Backward-compat: map old field names ──
            data = _normalize_legacy_fields(data)

            # ── Validate with Pydantic ──
            try:
                case = EvalCase.model_validate(data)
            except Exception as exc:
                result.errors.append({
                    "line": line_num,
                    "case_id": data.get("case_id", "?"),
                    "error": str(exc),
                })
                continue

            result.valid_lines += 1

            # ── Apply filters ──
            if not _passes_filters(
                case,
                tenant_id=tenant_id,
                endpoint_mode=endpoint_mode,
                case_id=case_id,
                slice_tag=slice_tag,
            ):
                result.skipped_by_filter += 1
                continue

            result.cases.append(case)

            if limit is not None and len(result.cases) >= limit:
                break

    return result


def validate_dataset(path: str | Path) -> DatasetLoadResult:
    """Validate dataset without filtering — useful for CI checks."""
    return load_dataset(path)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _normalize_legacy_fields(data: dict) -> dict:
    """Map old golden-set field names to 1.2A schema.

    Supports backward compatibility with the existing Phase 1
    golden_set.jsonl that uses `answer_type` instead of
    `expected_answer_type`.
    """
    # answer_type -> expected_answer_type
    if "answer_type" in data and "expected_answer_type" not in data:
        data["expected_answer_type"] = data.pop("answer_type")

    # Ensure slice_tags exists (old format might not have it)
    if "slice_tags" not in data:
        # Auto-generate from expected_answer_type
        eat = data.get("expected_answer_type", "")
        data["slice_tags"] = [eat] if eat else []

    # Ensure endpoint_mode exists (old format defaults to assistant)
    if "endpoint_mode" not in data:
        data["endpoint_mode"] = "assistant"

    # Ensure forbidden_keywords exists
    if "forbidden_keywords" not in data:
        data["forbidden_keywords"] = []

    # Ensure criticality exists
    if "criticality" not in data:
        data["criticality"] = "medium"

    return data


def _passes_filters(
    case: EvalCase,
    *,
    tenant_id: str | None,
    endpoint_mode: str | None,
    case_id: str | None,
    slice_tag: str | None,
) -> bool:
    """Check if a case passes all active filters."""
    if tenant_id is not None and case.tenant_id != tenant_id:
        return False

    if endpoint_mode is not None and case.endpoint_mode != endpoint_mode:
        return False

    if case_id is not None and case.case_id != case_id:
        return False

    if slice_tag is not None and slice_tag not in case.slice_tags:
        return False

    return True
