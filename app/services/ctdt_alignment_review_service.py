"""
Alignment Review Service — R6.2C Objective ↔ Outcome alignment review.

Read-only rule-based review: reads latest objective_update and outcome_update
drafts from ctdt_analysis_drafts, then evaluates alignment quality.

Guards:
  - No writes to Program / ProgramVersion / ProgramVersionRevision.
  - No writes to ctdt_analysis_drafts (read-only).
  - No LLM calls.
  - No raw file processing, no clean/chunk/embed.
  - No retrieval / context-pack.
  - Missing draft → missing_information, never crash.
"""
from __future__ import annotations

import logging
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.ctdt_analysis_draft_service import get_latest_analysis_draft

logger = logging.getLogger(__name__)


# ── Coverage / mapping status enums ──────────────────────────────────

_COVERAGE_STATUSES = frozenset(["covered", "partially_covered", "not_covered", "unknown"])
_MAPPING_STATUSES = frozenset(["valid", "weak", "missing", "unknown"])
_CONFIDENCE_LEVELS = frozenset(["low", "medium", "high"])
_GAP_TYPES = frozenset([
    "objective_not_covered", "outcome_unmapped", "low_confidence",
    "missing_evidence", "needs_human_review",
])
_SEVERITY_LEVELS = frozenset(["low", "medium", "high"])


# ── Main entry point ────────────────────────────────────────────────


async def review_objective_outcome_alignment(
    db: AsyncSession,
    *,
    tenant_id: str,
    update_cycle_id: str,
    program_code: str | None = None,
    program_id: str | None = None,
) -> dict[str, Any]:
    """
    Rule-based review of alignment between objective_update and outcome_update
    drafts.

    Returns a JSON-serialisable dict matching the R6.2C output schema.

    - Read-only: no DB writes, no LLM calls.
    - If either draft is missing: populates missing_information, returns
      zeroed summary, never crashes.
    """

    # ── Step 1: Read latest objective_update draft ────────────────────
    obj_draft = await get_latest_analysis_draft(
        db,
        tenant_id=tenant_id,
        update_cycle_id=update_cycle_id,
        program_code=program_code,
        analysis_mode="design",
        draft_type="objective_update",
        status="draft",
    )

    # ── Step 2: Read latest outcome_update draft ─────────────────────
    out_draft = await get_latest_analysis_draft(
        db,
        tenant_id=tenant_id,
        update_cycle_id=update_cycle_id,
        program_code=program_code,
        analysis_mode="design",
        draft_type="outcome_update",
        status="draft",
    )

    # ── Step 3: Handle missing drafts ────────────────────────────────
    missing_information: list[dict[str, str]] = []

    if obj_draft is None:
        missing_information.append({
            "type": "objective_update",
            "description": "Chưa có draft mục tiêu đào tạo để review alignment.",
        })

    if out_draft is None:
        missing_information.append({
            "type": "outcome_update",
            "description": "Chưa có draft chuẩn đầu ra để review alignment.",
        })

    if obj_draft is None or out_draft is None:
        # Either draft missing → cannot evaluate alignment without both
        return _build_missing_draft_response(
            update_cycle_id=update_cycle_id,
            program_code=program_code,
            objective_draft=obj_draft,
            outcome_draft=out_draft,
            missing_information=missing_information,
        )

    # ── Step 4: Extract payloads ─────────────────────────────────────
    obj_payload = (obj_draft.result_payload or {}) if obj_draft else {}
    out_payload = (out_draft.result_payload or {}) if out_draft else {}

    proposed_objectives: list[dict] = obj_payload.get("proposed_objectives", [])
    proposed_outcomes: list[dict] = out_payload.get("proposed_outcomes", [])
    existing_alignment: list[dict] = out_payload.get("objective_outcome_alignment", [])

    # Build lookup: objective_code → objective dict
    obj_by_code: dict[str, dict] = {}
    for obj in proposed_objectives:
        code = obj.get("code") or obj.get("objective_code")
        if code:
            obj_by_code[code] = obj

    # ── Step 5: Rule-based review ────────────────────────────────────
    objective_coverage = _evaluate_objective_coverage(
        proposed_objectives, proposed_outcomes, obj_by_code,
    )
    outcome_mapping_quality = _evaluate_outcome_mapping_quality(
        proposed_outcomes, obj_by_code,
    )
    gaps = _collect_gaps(objective_coverage, outcome_mapping_quality)
    summary = _build_summary(
        proposed_objectives, proposed_outcomes,
        objective_coverage, outcome_mapping_quality,
    )
    next_actions = _build_next_actions(gaps, missing_information)

    return {
        "update_cycle_id": update_cycle_id,
        "program_code": program_code,
        "review_type": "objective_outcome_alignment",
        "objective_draft_id": obj_draft.id if obj_draft else None,
        "outcome_draft_id": out_draft.id if out_draft else None,
        "summary": summary,
        "objective_coverage": objective_coverage,
        "outcome_mapping_quality": outcome_mapping_quality,
        "gaps": gaps,
        "missing_information": missing_information,
        "next_actions": next_actions,
    }


# ── Objective coverage evaluation ────────────────────────────────────


def _evaluate_objective_coverage(
    proposed_objectives: list[dict],
    proposed_outcomes: list[dict],
    obj_by_code: dict[str, dict],
) -> list[dict[str, Any]]:
    """Evaluate how well each objective is covered by proposed outcomes."""

    # Build reverse map: objective_code → list of outcome codes that map to it
    obj_to_outcomes: dict[str, list[str]] = {}
    obj_to_outcome_details: dict[str, list[dict]] = {}

    for outcome in proposed_outcomes:
        outcome_code = outcome.get("code", "")
        mapped_objectives = outcome.get("mapped_objectives", [])
        if not isinstance(mapped_objectives, list):
            mapped_objectives = []

        for mapping in mapped_objectives:
            if isinstance(mapping, dict):
                obj_code = mapping.get("objective_code", "")
            elif isinstance(mapping, str):
                obj_code = mapping
            else:
                continue

            if obj_code:
                obj_to_outcomes.setdefault(obj_code, []).append(outcome_code)
                obj_to_outcome_details.setdefault(obj_code, []).append(outcome)

    coverage_results: list[dict[str, Any]] = []

    for obj in proposed_objectives:
        obj_code = obj.get("code") or obj.get("objective_code") or ""
        obj_content = obj.get("proposed_content") or obj.get("objective_content") or ""

        if not obj_code and not obj_content:
            coverage_results.append({
                "objective_code": obj_code,
                "objective_content": obj_content,
                "mapped_outcomes": [],
                "coverage_status": "unknown",
                "issues": ["missing_objective_code"],
                "recommendation": "Mục tiêu thiếu mã, khó đối chiếu với CĐR.",
            })
            continue

        mapped_outcome_codes = obj_to_outcomes.get(obj_code, [])
        mapped_outcome_dicts = obj_to_outcome_details.get(obj_code, [])
        issues: list[str] = []

        if not mapped_outcome_codes:
            coverage_status = "not_covered"
            issues.append("no_outcome_mapped")
            recommendation = (
                f"Mục tiêu {obj_code} chưa có CĐR nào liên kết. "
                "Cần bổ sung CĐR hoặc xác nhận mục tiêu này được phủ gián tiếp."
            )
        else:
            # Check quality of mapped outcomes
            has_low_confidence = False
            has_missing_evidence = False

            for out_dict in mapped_outcome_dicts:
                confidence = out_dict.get("confidence", "medium")
                quality_flags = out_dict.get("quality_flags", [])
                if not isinstance(quality_flags, list):
                    quality_flags = []

                if confidence == "low":
                    has_low_confidence = True
                if "missing_evidence" in quality_flags:
                    has_missing_evidence = True

            if has_low_confidence or has_missing_evidence:
                coverage_status = "partially_covered"
                if has_low_confidence:
                    issues.append("mapped_outcome_low_confidence")
                if has_missing_evidence:
                    issues.append("mapped_outcome_missing_evidence")
                recommendation = (
                    f"Mục tiêu {obj_code} có CĐR liên kết nhưng chất lượng "
                    "CĐR chưa đảm bảo (confidence thấp hoặc thiếu evidence)."
                )
            else:
                coverage_status = "covered"
                recommendation = ""

        coverage_results.append({
            "objective_code": obj_code,
            "objective_content": obj_content,
            "mapped_outcomes": mapped_outcome_codes,
            "coverage_status": coverage_status,
            "issues": issues,
            "recommendation": recommendation,
        })

    return coverage_results


# ── Outcome mapping quality evaluation ───────────────────────────────


def _evaluate_outcome_mapping_quality(
    proposed_outcomes: list[dict],
    obj_by_code: dict[str, dict],
) -> list[dict[str, Any]]:
    """Evaluate mapping quality of each proposed outcome."""

    quality_results: list[dict[str, Any]] = []

    for outcome in proposed_outcomes:
        outcome_code = outcome.get("code", "")
        outcome_content = outcome.get("proposed_content", "")
        mapped_objectives_raw = outcome.get("mapped_objectives", [])
        if not isinstance(mapped_objectives_raw, list):
            mapped_objectives_raw = []

        confidence = outcome.get("confidence", "medium")
        quality_flags = outcome.get("quality_flags", [])
        if not isinstance(quality_flags, list):
            quality_flags = []

        issues: list[str] = []

        # Extract objective codes from mapped_objectives
        mapped_obj_codes: list[str] = []
        for mapping in mapped_objectives_raw:
            if isinstance(mapping, dict):
                code = mapping.get("objective_code", "")
            elif isinstance(mapping, str):
                code = mapping
            else:
                continue
            if code:
                mapped_obj_codes.append(code)

        # Rule: mapped_objectives rỗng → missing
        if not mapped_obj_codes:
            mapping_status = "missing"
            issues.append("missing_objective_mapping")
        else:
            # Rule: mapped to objective_code not in objective draft
            orphan_codes = [c for c in mapped_obj_codes if c not in obj_by_code]
            if orphan_codes:
                mapping_status = "weak"
                issues.append("mapped_objective_not_found")
            else:
                mapping_status = "valid"

        # Rule: confidence low
        if confidence == "low":
            issues.append("low_confidence")

        # Rule: missing_evidence in quality_flags
        if "missing_evidence" in quality_flags:
            issues.append("missing_evidence")

        # Rule: needs_human_review in quality_flags
        if "needs_human_review" in quality_flags:
            issues.append("needs_human_review")

        # If issues exist but status is still valid, check if it should remain valid
        # (mapping_status is about the objective mapping, not about other quality issues)
        if not issues and mapping_status != "missing":
            mapping_status = "valid"

        # Normalize confidence for output
        conf_out = confidence if confidence in _CONFIDENCE_LEVELS else "medium"

        quality_results.append({
            "outcome_code": outcome_code,
            "outcome_content": outcome_content,
            "mapped_objectives": mapped_obj_codes,
            "mapping_status": mapping_status,
            "confidence": conf_out,
            "quality_flags": list(quality_flags),
            "issues": issues,
            "recommendation": _outcome_recommendation(outcome_code, issues),
        })

    return quality_results


def _outcome_recommendation(outcome_code: str, issues: list[str]) -> str:
    """Build a recommendation string for an outcome based on its issues."""
    if not issues:
        return ""

    parts: list[str] = []
    if "missing_objective_mapping" in issues:
        parts.append("Cần liên kết CĐR với mục tiêu đào tạo")
    if "mapped_objective_not_found" in issues:
        parts.append("Mục tiêu đào tạo được liên kết không tồn tại trong draft")
    if "low_confidence" in issues:
        parts.append("Độ tin cậy thấp, cần bổ sung căn cứ")
    if "missing_evidence" in issues:
        parts.append("Thiếu bằng chứng hỗ trợ")
    if "needs_human_review" in issues:
        parts.append("Cần hội đồng xem xét")

    prefix = f"CĐR {outcome_code}: " if outcome_code else ""
    return prefix + "; ".join(parts) + "."


# ── Gap collection ──────────────────────────────────────────────────


def _collect_gaps(
    objective_coverage: list[dict],
    outcome_mapping_quality: list[dict],
) -> list[dict[str, Any]]:
    """Collect gaps from coverage and mapping results."""
    gaps: list[dict[str, Any]] = []

    for cov in objective_coverage:
        if cov["coverage_status"] == "not_covered":
            gaps.append({
                "type": "objective_not_covered",
                "severity": "high",
                "description": (
                    f"Mục tiêu {cov['objective_code']} không có CĐR nào liên kết."
                ),
                "related_objective_code": cov["objective_code"],
                "related_outcome_code": None,
            })

    for qual in outcome_mapping_quality:
        if qual["mapping_status"] == "missing":
            gaps.append({
                "type": "outcome_unmapped",
                "severity": "high",
                "description": (
                    f"CĐR {qual['outcome_code']} không liên kết với mục tiêu đào tạo nào."
                ),
                "related_objective_code": None,
                "related_outcome_code": qual["outcome_code"],
            })

        if "low_confidence" in qual["issues"]:
            gaps.append({
                "type": "low_confidence",
                "severity": "medium",
                "description": (
                    f"CĐR {qual['outcome_code']} có độ tin cậy thấp."
                ),
                "related_objective_code": None,
                "related_outcome_code": qual["outcome_code"],
            })

        if "missing_evidence" in qual["issues"]:
            gaps.append({
                "type": "missing_evidence",
                "severity": "medium",
                "description": (
                    f"CĐR {qual['outcome_code']} thiếu bằng chứng hỗ trợ."
                ),
                "related_objective_code": None,
                "related_outcome_code": qual["outcome_code"],
            })

        if "needs_human_review" in qual["issues"]:
            gaps.append({
                "type": "needs_human_review",
                "severity": "medium",
                "description": (
                    f"CĐR {qual['outcome_code']} cần hội đồng xem xét."
                ),
                "related_objective_code": None,
                "related_outcome_code": qual["outcome_code"],
            })

    return gaps


# ── Summary builder ─────────────────────────────────────────────────


def _build_summary(
    proposed_objectives: list[dict],
    proposed_outcomes: list[dict],
    objective_coverage: list[dict],
    outcome_mapping_quality: list[dict],
) -> dict[str, int]:
    """Build summary counts."""
    covered = sum(1 for c in objective_coverage if c["coverage_status"] == "covered")
    partially = sum(1 for c in objective_coverage if c["coverage_status"] == "partially_covered")
    not_covered = sum(1 for c in objective_coverage if c["coverage_status"] == "not_covered")
    unmapped = sum(1 for q in outcome_mapping_quality if q["mapping_status"] == "missing")
    low_conf = sum(1 for q in outcome_mapping_quality if "low_confidence" in q["issues"])
    needs_hr = sum(1 for q in outcome_mapping_quality if "needs_human_review" in q["issues"])

    return {
        "objectives_count": len(proposed_objectives),
        "outcomes_count": len(proposed_outcomes),
        "covered_objectives_count": covered,
        "partially_covered_objectives_count": partially,
        "not_covered_objectives_count": not_covered,
        "unmapped_outcomes_count": unmapped,
        "low_confidence_outcomes_count": low_conf,
        "needs_human_review_count": needs_hr,
    }


# ── Next actions builder ────────────────────────────────────────────


def _build_next_actions(
    gaps: list[dict],
    missing_information: list[dict],
) -> list[dict[str, str]]:
    """Build next-action recommendations."""
    actions: list[dict[str, str]] = []

    # If missing drafts, priority action is to create them
    for mi in missing_information:
        mi_type = mi.get("type", "")
        if mi_type == "objective_update":
            actions.append({
                "action": "Tạo draft mục tiêu đào tạo (R6.1B) trước khi review alignment.",
                "owner_hint": "admin",
                "priority": "high",
            })
        elif mi_type == "outcome_update":
            actions.append({
                "action": "Tạo draft chuẩn đầu ra (R6.2B) trước khi review alignment.",
                "owner_hint": "admin",
                "priority": "high",
            })

    # Gap-based actions
    has_not_covered = any(g["type"] == "objective_not_covered" for g in gaps)
    has_unmapped = any(g["type"] == "outcome_unmapped" for g in gaps)
    has_needs_hr = any(g["type"] == "needs_human_review" for g in gaps)

    if has_not_covered:
        actions.append({
            "action": "Bổ sung CĐR cho các mục tiêu chưa được phủ.",
            "owner_hint": "bo_mon",
            "priority": "high",
        })

    if has_unmapped:
        actions.append({
            "action": "Liên kết CĐR chưa có mục tiêu với mục tiêu đào tạo phù hợp.",
            "owner_hint": "bo_mon",
            "priority": "high",
        })

    if has_needs_hr:
        actions.append({
            "action": "Trình hội đồng xem xét các CĐR cần review.",
            "owner_hint": "hoi_dong",
            "priority": "medium",
        })

    if not gaps and not missing_information:
        actions.append({
            "action": "Alignment mục tiêu ↔ CĐR đạt yêu cầu. Có thể chuyển sang bước course/matrix.",
            "owner_hint": "khoa",
            "priority": "low",
        })

    return actions


# ── Response for missing draft(s) ────────────────────────────────────


def _build_missing_draft_response(
    *,
    update_cycle_id: str,
    program_code: str | None,
    objective_draft,
    outcome_draft,
    missing_information: list[dict],
) -> dict[str, Any]:
    """Return a valid response when one or both drafts are missing.

    Counts objectives/outcomes from whichever draft exists so the summary
    reflects available data.  No coverage/mapping/gaps are evaluated —
    that would produce false positives.
    """
    obj_payload = (objective_draft.result_payload or {}) if objective_draft else {}
    out_payload = (outcome_draft.result_payload or {}) if outcome_draft else {}

    objectives_count = len(obj_payload.get("proposed_objectives", []))
    outcomes_count = len(out_payload.get("proposed_outcomes", []))

    return {
        "update_cycle_id": update_cycle_id,
        "program_code": program_code,
        "review_type": "objective_outcome_alignment",
        "objective_draft_id": objective_draft.id if objective_draft else None,
        "outcome_draft_id": outcome_draft.id if outcome_draft else None,
        "summary": {
            "objectives_count": objectives_count,
            "outcomes_count": outcomes_count,
            "covered_objectives_count": 0,
            "partially_covered_objectives_count": 0,
            "not_covered_objectives_count": 0,
            "unmapped_outcomes_count": 0,
            "low_confidence_outcomes_count": 0,
            "needs_human_review_count": 0,
        },
        "objective_coverage": [],
        "outcome_mapping_quality": [],
        "gaps": [],
        "missing_information": missing_information,
        "next_actions": _build_next_actions([], missing_information),
    }
