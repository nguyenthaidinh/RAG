"""
Persistence helpers for CTDT analysis drafts.

This store is intentionally separate from official curriculum/program tables.
It only saves generated RAG analysis drafts such as Mau 06 change proposals.
"""
from __future__ import annotations

from dataclasses import asdict

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.ctdt_analysis_draft import CTDTAnalysisDraft
from app.services.ctdt_analysis_service import AnalysisCycleResult


DEFAULT_DRAFT_TYPE = "update_cycle_analysis"


def _result_payload_to_json(result: AnalysisCycleResult) -> dict:
    """Convert AnalysisCycleResult.result_payload into JSONB-safe dicts."""
    return {
        key: [asdict(item) for item in items]
        for key, items in result.result_payload.items()
    }


def _source_summary_to_json(result: AnalysisCycleResult) -> dict:
    """Convert AnalysisCycleResult.source_summary into a JSONB-safe dict."""
    return asdict(result.source_summary)


async def save_analysis_draft(
    db: AsyncSession,
    *,
    tenant_id: str,
    user_id: int | None,
    result: AnalysisCycleResult,
    program_id: str | None = None,
    draft_type: str = DEFAULT_DRAFT_TYPE,
    status: str = "draft",
) -> CTDTAnalysisDraft:
    """
    Persist an analysis result as a draft row.

    The caller owns the transaction. This function flushes so the draft ID is
    available, but does not commit.
    """
    draft = CTDTAnalysisDraft(
        tenant_id=tenant_id,
        update_cycle_id=result.update_cycle_id,
        program_id=program_id,
        program_code=result.program_code,
        program_name=result.program_name,
        analysis_mode=result.analysis_mode,
        draft_type=draft_type,
        result_payload=_result_payload_to_json(result),
        source_summary=_source_summary_to_json(result),
        created_by=user_id,
        updated_by=user_id,
        status=status,
    )
    db.add(draft)
    await db.flush()
    await db.refresh(draft)
    return draft


async def get_latest_analysis_draft(
    db: AsyncSession,
    *,
    tenant_id: str,
    update_cycle_id: str,
    program_id: str | None = None,
    program_code: str | None = None,
    analysis_mode: str = "draft",
    draft_type: str = DEFAULT_DRAFT_TYPE,
    status: str = "draft",
) -> CTDTAnalysisDraft | None:
    """Return the latest analysis draft scoped by tenant and update cycle."""
    stmt = select(CTDTAnalysisDraft).where(
        CTDTAnalysisDraft.tenant_id == tenant_id,
        CTDTAnalysisDraft.update_cycle_id == update_cycle_id,
        CTDTAnalysisDraft.analysis_mode == analysis_mode,
        CTDTAnalysisDraft.draft_type == draft_type,
        CTDTAnalysisDraft.status == status,
    )
    if program_id is not None:
        stmt = stmt.where(CTDTAnalysisDraft.program_id == program_id)
    if program_code is not None:
        stmt = stmt.where(CTDTAnalysisDraft.program_code == program_code)

    stmt = stmt.order_by(
        CTDTAnalysisDraft.updated_at.desc(),
        CTDTAnalysisDraft.id.desc(),
    ).limit(1)

    result = await db.execute(stmt)
    return result.scalars().first()


async def list_analysis_drafts(
    db: AsyncSession,
    *,
    tenant_id: str,
    update_cycle_id: str | None = None,
    program_code: str | None = None,
    analysis_mode: str | None = None,
    draft_type: str = DEFAULT_DRAFT_TYPE,
    status: str = "draft",
    limit: int = 20,
) -> list[CTDTAnalysisDraft]:
    """List analysis drafts scoped by tenant.

    By default only returns status='draft' rows. Pass status=None to include
    all statuses (e.g. archived).
    """
    stmt = select(CTDTAnalysisDraft).where(
        CTDTAnalysisDraft.tenant_id == tenant_id,
        CTDTAnalysisDraft.draft_type == draft_type,
    )
    if status is not None:
        stmt = stmt.where(CTDTAnalysisDraft.status == status)
    if update_cycle_id is not None:
        stmt = stmt.where(CTDTAnalysisDraft.update_cycle_id == update_cycle_id)
    if program_code is not None:
        stmt = stmt.where(CTDTAnalysisDraft.program_code == program_code)
    if analysis_mode is not None:
        stmt = stmt.where(CTDTAnalysisDraft.analysis_mode == analysis_mode)

    stmt = stmt.order_by(
        CTDTAnalysisDraft.updated_at.desc(),
        CTDTAnalysisDraft.id.desc(),
    ).limit(limit)

    result = await db.execute(stmt)
    return list(result.scalars().all())


async def save_raw_analysis_draft(
    db: AsyncSession,
    *,
    tenant_id: str,
    user_id: int | None,
    update_cycle_id: str,
    program_id: str | None = None,
    program_code: str | None = None,
    program_name: str | None = None,
    analysis_mode: str = "design",
    draft_type: str,
    result_payload: dict,
    source_summary: dict,
    status: str = "draft",
) -> CTDTAnalysisDraft:
    """Persist a raw dict payload as a draft row.

    Unlike ``save_analysis_draft`` this accepts plain dicts instead of
    ``AnalysisCycleResult``, making it usable for mapping drafts and other
    payloads that are not tied to the R5 analysis pipeline.

    The caller owns the transaction — this flushes but does NOT commit.
    """
    draft = CTDTAnalysisDraft(
        tenant_id=tenant_id,
        update_cycle_id=update_cycle_id,
        program_id=program_id,
        program_code=program_code,
        program_name=program_name,
        analysis_mode=analysis_mode,
        draft_type=draft_type,
        result_payload=result_payload,
        source_summary=source_summary,
        created_by=user_id,
        updated_by=user_id,
        status=status,
    )
    db.add(draft)
    await db.flush()
    await db.refresh(draft)
    return draft
