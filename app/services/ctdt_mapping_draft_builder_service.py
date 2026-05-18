"""
Mapping Draft Builder Service — R6.3B V1.

Builds objective↔outcome mapping draft from latest objective_update and
outcome_update drafts.  V1 only builds objective_outcome rows; course
mappings deferred to later phases.

Guards:
  - No LLM calls.
  - No retrieval / context-pack.
  - No file processing.
  - No Program / ProgramVersion / ProgramVersionRevision writes.
  - No official mapping writes.
  - Missing draft → missing_information + empty rows, never crash.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.ctdt_analysis_draft_service import (
    get_latest_analysis_draft,
    save_raw_analysis_draft,
)
from app.services.ctdt_mapping_draft_contract import (
    MappingDraftPayload,
    MappingDraftType,
    MappingSourceRef,
    MappingSourceType,
    MappingStatus,
    ObjectiveOutcomeMappingRow,
    dedupe_warnings,
    normalize_confidence,
)

logger = logging.getLogger(__name__)

_V1_WARNING = (
    "R6.3B V1 only builds objective_outcome mapping. "
    "Course mappings will be handled in a later phase."
)

_V1_SUPPORTED = frozenset([MappingDraftType.OBJECTIVE_OUTCOME])
_V1_DEFERRED = MappingDraftType.ALL - _V1_SUPPORTED


# ── Exceptions ───────────────────────────────────────────────────────


class MappingDraftSaveError(RuntimeError):
    """Raised when saving a mapping draft to DB fails."""


# ── Result dataclass ─────────────────────────────────────────────────


@dataclass
class MappingDraftBuildResult:
    update_cycle_id: str = ""
    program_id: str | None = None
    program_code: str | None = None
    program_name: str | None = None
    draft_type: str = "mapping_draft"
    payload: dict = field(default_factory=dict)
    source_summary: dict = field(default_factory=dict)
    missing_information: list[dict[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    draft_id: int | None = None
    draft_saved: bool = False


# ── Main entry point ─────────────────────────────────────────────────


async def build_mapping_draft(
    db: AsyncSession,
    *,
    tenant_id: str,
    user_id: int | None,
    update_cycle_id: str,
    program_id: str | None = None,
    program_code: str | None = None,
    program_name: str | None = None,
    save_draft: bool = False,
    mapping_types: list[str] | None = None,
) -> MappingDraftBuildResult:
    """Build V1 mapping draft (objective_outcome only).

    Reads latest objective_update and outcome_update drafts, then
    produces ObjectiveOutcomeMappingRow entries from mapped_objectives.

    Returns ``MappingDraftBuildResult``.

    Raises ``MappingDraftSaveError`` if save_draft=True and persistence
    fails.
    """

    # ── Validate mapping_types ───────────────────────────────────────
    if not mapping_types:
        mapping_types = [MappingDraftType.OBJECTIVE_OUTCOME]

    missing_information: list[dict[str, str]] = []
    warnings: list[str] = []

    effective_types: list[str] = []
    for mt in mapping_types:
        if mt not in MappingDraftType.ALL:
            warnings.append(f"Unsupported mapping_type '{mt}' ignored.")
        elif mt in _V1_DEFERRED:
            warnings.append(
                f"Mapping type '{mt}' is deferred in R6.3B V1."
            )
        else:
            effective_types.append(mt)

    # ── Read latest drafts ───────────────────────────────────────────
    obj_draft = await get_latest_analysis_draft(
        db,
        tenant_id=tenant_id,
        update_cycle_id=update_cycle_id,
        program_id=program_id,
        program_code=program_code,
        analysis_mode="design",
        draft_type="objective_update",
        status="draft",
    )

    out_draft = await get_latest_analysis_draft(
        db,
        tenant_id=tenant_id,
        update_cycle_id=update_cycle_id,
        program_id=program_id,
        program_code=program_code,
        analysis_mode="design",
        draft_type="outcome_update",
        status="draft",
    )

    if obj_draft is None:
        missing_information.append({
            "type": "objective_update",
            "description": "Chưa có draft mục tiêu đào tạo để build mapping mục tiêu ↔ CĐR.",
        })

    if out_draft is None:
        missing_information.append({
            "type": "outcome_update",
            "description": "Chưa có draft chuẩn đầu ra để build mapping mục tiêu ↔ CĐR.",
        })

    # ── Build payload ────────────────────────────────────────────────
    payload = MappingDraftPayload(
        update_cycle_id=update_cycle_id,
        program_id=program_id,
        program_code=program_code,
        program_name=program_name,
    )

    if obj_draft is not None and out_draft is not None:
        if MappingDraftType.OBJECTIVE_OUTCOME in effective_types:
            obj_payload = obj_draft.result_payload or {}
            out_payload = out_draft.result_payload or {}
            rows, dupe_warnings = _build_objective_outcome_rows(
                proposed_objectives=obj_payload.get("proposed_objectives", []),
                proposed_outcomes=out_payload.get("proposed_outcomes", []),
                update_cycle_id=update_cycle_id,
                program_code=program_code,
                program_id=program_id,
            )
            payload.objective_outcome_rows = rows
            warnings.extend(dupe_warnings)

    # V1 warning
    warnings.append(_V1_WARNING)
    payload.warnings = list(warnings)

    # Source summary
    source_summary = payload.build_source_summary()

    # ── Optional save ────────────────────────────────────────────────
    draft_id: int | None = None
    draft_saved = False

    if save_draft:
        has_supported = bool(effective_types)
        if missing_information:
            warnings.append(
                "Mapping draft was not saved because required "
                "objective/outcome draft is missing."
            )
        elif not has_supported:
            warnings.append(
                "Mapping draft was not saved because no supported "
                "mapping type was requested."
            )
        else:
            try:
                draft_row = await save_raw_analysis_draft(
                    db,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    update_cycle_id=update_cycle_id,
                    program_id=program_id,
                    program_code=program_code,
                    program_name=program_name,
                    analysis_mode="design",
                    draft_type="mapping_draft",
                    result_payload=payload.to_dict(),
                    source_summary=source_summary,
                    status="draft",
                )
                await db.commit()
                draft_id = draft_row.id
                draft_saved = True
            except Exception as exc:
                await db.rollback()
                logger.exception(
                    "mapping_draft.save_failed tenant=%s cycle=%s",
                    tenant_id, update_cycle_id,
                )
                raise MappingDraftSaveError(
                    "Không lưu được mapping draft."
                ) from exc

    return MappingDraftBuildResult(
        update_cycle_id=update_cycle_id,
        program_id=program_id,
        program_code=program_code,
        program_name=program_name,
        payload=payload.to_dict(),
        source_summary=source_summary,
        missing_information=missing_information,
        warnings=dedupe_warnings(warnings),
        draft_id=draft_id,
        draft_saved=draft_saved,
    )


# ── Row builder ──────────────────────────────────────────────────────


def _build_objective_outcome_rows(
    *,
    proposed_objectives: list[dict],
    proposed_outcomes: list[dict],
    update_cycle_id: str,
    program_code: str | None,
    program_id: str | None,
) -> tuple[list[ObjectiveOutcomeMappingRow], list[str]]:
    """Build ObjectiveOutcomeMappingRow from draft payloads.

    Returns (rows, payload_warnings).  Duplicate (obj, outcome) pairs
    are deduplicated — only the first occurrence is kept.
    """

    obj_by_code: dict[str, dict] = {}
    for obj in proposed_objectives:
        code = obj.get("code") or obj.get("objective_code")
        if code:
            obj_by_code[code] = obj

    rows: list[ObjectiveOutcomeMappingRow] = []
    seen_pairs: set[tuple[str, str]] = set()
    payload_warnings: list[str] = []

    for outcome in proposed_outcomes:
        outcome_code = outcome.get("code", "")
        outcome_content = outcome.get("proposed_content", "")
        confidence_raw = outcome.get("confidence", "medium")
        evidence_refs = outcome.get("evidence_refs", [])
        if not isinstance(evidence_refs, list):
            evidence_refs = []

        mapped_objectives = outcome.get("mapped_objectives", [])
        if not isinstance(mapped_objectives, list):
            mapped_objectives = []

        for mapping in mapped_objectives:
            if isinstance(mapping, dict):
                obj_code = mapping.get("objective_code", "")
                mapping_reason = mapping.get("mapping_reason", "")
            elif isinstance(mapping, str):
                obj_code = mapping
                mapping_reason = ""
            else:
                continue

            if not obj_code:
                continue

            # ── Dedupe ───────────────────────────────────────────
            pair = (obj_code, outcome_code)
            if pair in seen_pairs:
                payload_warnings.append(
                    f"Duplicate objective_outcome mapping "
                    f"{obj_code}→{outcome_code} skipped."
                )
                continue
            seen_pairs.add(pair)

            row_warnings: list[str] = []

            # Lookup objective
            obj_dict = obj_by_code.get(obj_code)
            if obj_dict is not None:
                obj_content = (
                    obj_dict.get("proposed_content")
                    or obj_dict.get("objective_content")
                    or ""
                )
                status = MappingStatus.DRAFT
                confidence = normalize_confidence(confidence_raw)
            else:
                obj_content = ""
                status = MappingStatus.NEEDS_REVIEW
                confidence = "low"
                row_warnings.append("mapped_objective_not_found")

            # Contribution level: outcome draft only indicates mapping exists,
            # not the actual level → default to 1 (low)
            contribution_level = 1
            row_warnings.append(
                "No explicit contribution level in outcome draft; "
                "defaulted to 1 for draft review."
            )

            # Source refs from evidence_refs — never fabricate
            source_refs = _extract_source_refs(
                evidence_refs,
                update_cycle_id=update_cycle_id,
                program_code=program_code,
                program_id=program_id,
            )

            rows.append(ObjectiveOutcomeMappingRow(
                objective_code=obj_code,
                objective_content=obj_content,
                outcome_code=outcome_code,
                outcome_content=outcome_content,
                contribution_level=contribution_level,
                rationale=mapping_reason or None,
                confidence=confidence,
                status=status,
                source_type=MappingSourceType.GENERATED_FROM_DRAFT,
                source_refs=source_refs,
                warnings=dedupe_warnings(row_warnings),
            ))

    return rows, payload_warnings


def _extract_source_refs(
    evidence_refs: list,
    *,
    update_cycle_id: str,
    program_code: str | None,
    program_id: str | None,
) -> list[MappingSourceRef]:
    """Convert evidence_refs to MappingSourceRef list.

    Skips entries without ai_document_id to avoid fabricated sources.
    """
    refs: list[MappingSourceRef] = []
    for ev in evidence_refs:
        if not isinstance(ev, dict):
            continue
        aid = ev.get("ai_document_id")
        if aid is None:
            continue
        refs.append(MappingSourceRef(
            ai_document_id=aid,
            external_file_id=ev.get("external_file_id"),
            filename=ev.get("filename"),
            document_role=ev.get("document_role"),
            chunk_id=ev.get("chunk_id"),
            chunk_index=ev.get("chunk_index"),
            score=ev.get("score"),
            quote=ev.get("quote") or ev.get("text"),
            update_cycle_id=update_cycle_id,
            program_code=program_code,
            program_id=program_id,
        ))
    return refs
