"""
Admin Source Platform API (Phase 4 + Phase 7 + Observability/Investigation/Action/Repair).

Endpoints for managing onboarded knowledge sources, triggering syncs,
managing schedules, viewing health status, observability, investigation,
single-item sync, and link-centric repair.

Endpoints:
  ── Base Control-Plane ──
  GET    /api/v1/admin/source-platform/sources
  GET    /api/v1/admin/source-platform/sources/{id}
  POST   /api/v1/admin/source-platform/sources
  PUT    /api/v1/admin/source-platform/sources/{id}
  POST   /api/v1/admin/source-platform/sources/{id}/sync
  GET    /api/v1/admin/source-platform/sources/{id}/sync-runs
  PATCH  /api/v1/admin/source-platform/sources/{id}/schedule     (Phase 7)
  POST   /api/v1/admin/source-platform/sources/{id}/pause        (Phase 7)
  POST   /api/v1/admin/source-platform/sources/{id}/resume       (Phase 7)
  GET    /api/v1/admin/source-platform/sources/{id}/health        (Phase 7)

  ── Phase 1: Observability ──
  GET    /api/v1/admin/source-platform/sources/{id}/overview
  GET    /api/v1/admin/source-platform/sources/{id}/links
  GET    /api/v1/admin/source-platform/sources/{id}/links/{link_id}
  GET    /api/v1/admin/source-platform/sources/{id}/sync-runs/{run_id}

  ── Phase 2A: Investigation ──
  POST   /api/v1/admin/source-platform/sources/{id}/test-connection
  GET    /api/v1/admin/source-platform/sources/{id}/preview-refs
  GET    /api/v1/admin/source-platform/sources/{id}/items/{external_id}/preview

  ── Phase 2B: Action ──
  POST   /api/v1/admin/source-platform/sources/{id}/sync-one

  ── Phase 2C: Repair ──
  POST   /api/v1/admin/source-platform/sources/{id}/links/{link_id}/repair

RBAC:
  system_admin  → can view/act all tenants
  tenant_admin  → forced to own tenant
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth_deps import require_admin
from app.core.rbac import is_system_admin
from app.db.models.user import User
from app.db.session import get_db
from app.schemas.source_platform import (
    SourceConnectionTestResponse,
    SourceCreateRequest,
    SourceHealthResponse,
    SourceItemPreviewResponse,
    SourceLinkDetailResponse,
    SourceLinkListResponse,
    SourceListResponse,
    SourceOverviewResponse,
    SourcePreviewRefsResponse,
    SourceRepairRequest,
    SourceRepairResponse,
    SourceResponse,
    SourceScheduleUpdateRequest,
    SourceUpdateRequest,
    SyncOneRequest,
    SyncOneResponse,
    SyncRunDetailResponse,
    SyncRunListResponse,
    SyncTriggerResponse,
)
from app.services.source_platform.platform_admin_service import (
    PlatformAdminService,
)
from app.services.source_platform.repair_service import (
    SourcePlatformRepairService,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/admin/source-platform",
    tags=["admin-source-platform"],
)

_svc = PlatformAdminService()
_repair_svc = SourcePlatformRepairService()


# ── Lazy-init for phase services ─────────────────────────────────────

_obs_svc = None
_inv_svc = None
_action_svc = None


def _get_obs_svc():
    """Lazy-init observability service (Phase 1)."""
    global _obs_svc
    if _obs_svc is None:
        from app.services.source_platform.observability_service import (
            SourcePlatformObservabilityService,
        )
        _obs_svc = SourcePlatformObservabilityService()
    return _obs_svc


def _get_inv_svc():
    """Lazy-init investigation service (Phase 2A)."""
    global _inv_svc
    if _inv_svc is None:
        from app.services.source_platform.investigation_service import (
            SourcePlatformInvestigationService,
        )
        _inv_svc = SourcePlatformInvestigationService()
    return _inv_svc


def _get_action_svc():
    """Lazy-init action service (Phase 2B)."""
    global _action_svc
    if _action_svc is None:
        from app.services.source_platform.action_service import (
            SourcePlatformActionService,
        )
        _action_svc = SourcePlatformActionService()
    return _action_svc


# ── Helpers ──────────────────────────────────────────────────────────


def _scope_tenant(admin: User, tenant_id_param: str | None) -> str:
    """Resolve effective tenant_id based on RBAC."""
    if is_system_admin(admin.role):
        return tenant_id_param or admin.tenant_id
    return admin.tenant_id


# ── Source CRUD Endpoints ────────────────────────────────────────────


@router.get("/sources", response_model=SourceListResponse)
async def list_sources(
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
    tenant_id: str | None = Query(None),
    connector_type: str | None = Query(None),
    is_active: bool | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
):
    """
    List onboarded sources.

    🔒 Admin-only.
      - system_admin: can filter by tenant_id
      - tenant_admin: forced to own tenant
    """
    scoped_tenant = _scope_tenant(admin, tenant_id)

    items, total = await _svc.list_sources(
        db,
        tenant_id=scoped_tenant,
        connector_type=connector_type,
        is_active=is_active,
        page=page,
        page_size=page_size,
    )

    return SourceListResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/sources/{source_id}", response_model=SourceResponse)
async def get_source(
    source_id: int,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
    tenant_id: str | None = Query(None),
):
    """
    Get source detail.

    🔒 Admin-only, tenant-scoped.
    """
    scoped_tenant = _scope_tenant(admin, tenant_id)

    source = await _svc.get_source(
        db, tenant_id=scoped_tenant, source_id=source_id
    )
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")
    return source


@router.post("/sources", response_model=SourceResponse, status_code=201)
async def create_source(
    body: SourceCreateRequest,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
    tenant_id: str | None = Query(None),
):
    """
    Create a new onboarded source.

    🔒 Admin-only, tenant-scoped.
    """
    scoped_tenant = _scope_tenant(admin, tenant_id)

    try:
        source = await _svc.create_source(
            db,
            tenant_id=scoped_tenant,
            data=body.model_dump(exclude_unset=False),
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    return source


@router.put("/sources/{source_id}", response_model=SourceResponse)
async def update_source(
    source_id: int,
    body: SourceUpdateRequest,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
    tenant_id: str | None = Query(None),
):
    """
    Update an onboarded source.

    🔒 Admin-only, tenant-scoped.
    """
    scoped_tenant = _scope_tenant(admin, tenant_id)

    source = await _svc.update_source(
        db,
        tenant_id=scoped_tenant,
        source_id=source_id,
        data=body.model_dump(exclude_unset=True),
    )
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")
    return source


# ── Sync Endpoints ───────────────────────────────────────────────────


@router.post(
    "/sources/{source_id}/sync",
    response_model=SyncTriggerResponse,
)
async def trigger_sync(
    source_id: int,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
    tenant_id: str | None = Query(None),
):
    """
    Trigger a sync for one source.

    Uses coordinator lock guard — rejects if source already has
    a running sync (Phase 7).

    🔒 Admin-only, tenant-scoped.
    """
    scoped_tenant = _scope_tenant(admin, tenant_id)

    try:
        result = await _svc.trigger_sync_coordinated(
            db,
            tenant_id=scoped_tenant,
            source_id=source_id,
            triggered_by="api",
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    if result is None:
        raise HTTPException(status_code=404, detail="Source not found")

    return SyncTriggerResponse(**result)


@router.get(
    "/sources/{source_id}/sync-runs",
    response_model=SyncRunListResponse,
)
async def list_sync_runs(
    source_id: int,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
    tenant_id: str | None = Query(None),
    limit: int = Query(20, ge=1, le=100),
):
    """
    List sync run history for a source.

    🔒 Admin-only, tenant-scoped.
    """
    scoped_tenant = _scope_tenant(admin, tenant_id)

    result = await _svc.list_sync_runs(
        db,
        tenant_id=scoped_tenant,
        source_id=source_id,
        limit=limit,
    )
    if result is None:
        raise HTTPException(status_code=404, detail="Source not found")

    items, total = result

    return SyncRunListResponse(items=items, total=total)


# ── Schedule Endpoints (Phase 7) ─────────────────────────────────────


@router.patch(
    "/sources/{source_id}/schedule",
    response_model=SourceResponse,
)
async def update_schedule(
    source_id: int,
    body: SourceScheduleUpdateRequest,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
    tenant_id: str | None = Query(None),
):
    """
    Update sync schedule for a source.

    🔒 Admin-only, tenant-scoped.
    """
    scoped_tenant = _scope_tenant(admin, tenant_id)

    source = await _svc.update_schedule(
        db,
        tenant_id=scoped_tenant,
        source_id=source_id,
        data=body.model_dump(exclude_unset=True),
    )
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")
    return source


@router.post(
    "/sources/{source_id}/pause",
    response_model=SourceResponse,
)
async def pause_source(
    source_id: int,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
    tenant_id: str | None = Query(None),
):
    """
    Pause scheduled sync for a source (sets sync_enabled=false).

    🔒 Admin-only, tenant-scoped.
    """
    scoped_tenant = _scope_tenant(admin, tenant_id)

    source = await _svc.pause_source(
        db, tenant_id=scoped_tenant, source_id=source_id
    )
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")
    return source


@router.post(
    "/sources/{source_id}/resume",
    response_model=SourceResponse,
)
async def resume_source(
    source_id: int,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
    tenant_id: str | None = Query(None),
):
    """
    Resume scheduled sync for a source (sets sync_enabled=true).

    🔒 Admin-only, tenant-scoped.
    """
    scoped_tenant = _scope_tenant(admin, tenant_id)

    source = await _svc.resume_source(
        db, tenant_id=scoped_tenant, source_id=source_id
    )
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")
    return source


# ── Health Endpoint (Phase 7) ────────────────────────────────────────


@router.get(
    "/sources/{source_id}/health",
    response_model=SourceHealthResponse,
)
async def get_source_health(
    source_id: int,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
    tenant_id: str | None = Query(None),
):
    """
    Get health status for a source.

    Returns sync health, failure streaks, and schedule info.

    🔒 Admin-only, tenant-scoped.
    """
    scoped_tenant = _scope_tenant(admin, tenant_id)

    health = await _svc.get_health(
        db, tenant_id=scoped_tenant, source_id=source_id
    )
    if not health:
        raise HTTPException(status_code=404, detail="Source not found")
    return health


# ── Phase 1: Observability Endpoints ─────────────────────────────────


@router.get(
    "/sources/{source_id}/overview",
    response_model=SourceOverviewResponse,
)
async def get_source_overview(
    source_id: int,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
    tenant_id: str | None = Query(None),
):
    """
    Get control-room overview for a source.

    Returns link counts, attention flags, and operational status.

    🔒 Admin-only, tenant-scoped.
    """
    scoped_tenant = _scope_tenant(admin, tenant_id)

    overview = await _get_obs_svc().get_source_overview(
        db, tenant_id=scoped_tenant, source_id=source_id
    )
    if not overview:
        raise HTTPException(status_code=404, detail="Source not found")
    return overview


@router.get(
    "/sources/{source_id}/links",
    response_model=SourceLinkListResponse,
)
async def list_source_links(
    source_id: int,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
    tenant_id: str | None = Query(None),
    status: str | None = Query(None),
    q: str | None = Query(None, description="Search external_id"),
    document_id: int | None = Query(None),
    has_document: bool | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    sort_by: str = Query("last_seen_at"),
    sort_order: str = Query("desc"),
):
    """
    List source document links with filtering and pagination.

    🔒 Admin-only, tenant-scoped.
    """
    scoped_tenant = _scope_tenant(admin, tenant_id)

    result = await _get_obs_svc().list_source_links(
        db,
        tenant_id=scoped_tenant,
        source_id=source_id,
        status=status,
        q=q,
        document_id=document_id,
        has_document=has_document,
        page=page,
        page_size=page_size,
        sort_by=sort_by,
        sort_order=sort_order,
    )
    if result is None:
        raise HTTPException(status_code=404, detail="Source not found")
    return result


@router.get(
    "/sources/{source_id}/links/{link_id}",
    response_model=SourceLinkDetailResponse,
)
async def get_source_link_detail(
    source_id: int,
    link_id: int,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
    tenant_id: str | None = Query(None),
):
    """
    Get detailed view of a single source document link.

    🔒 Admin-only, tenant-scoped.
    """
    scoped_tenant = _scope_tenant(admin, tenant_id)

    detail = await _get_obs_svc().get_source_link_detail(
        db,
        tenant_id=scoped_tenant,
        source_id=source_id,
        link_id=link_id,
    )
    if detail is None:
        raise HTTPException(
            status_code=404, detail="Source or link not found"
        )
    return detail


@router.get(
    "/sources/{source_id}/sync-runs/{run_id}",
    response_model=SyncRunDetailResponse,
)
async def get_sync_run_detail(
    source_id: int,
    run_id: int,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
    tenant_id: str | None = Query(None),
):
    """
    Get detailed view of a single sync run with derived outcome.

    🔒 Admin-only, tenant-scoped.
    """
    scoped_tenant = _scope_tenant(admin, tenant_id)

    detail = await _get_obs_svc().get_sync_run_detail(
        db,
        tenant_id=scoped_tenant,
        source_id=source_id,
        run_id=run_id,
    )
    if detail is None:
        raise HTTPException(
            status_code=404, detail="Source or sync run not found"
        )
    return detail


# ── Phase 2A: Investigation Endpoints ────────────────────────────────


@router.post(
    "/sources/{source_id}/test-connection",
    response_model=SourceConnectionTestResponse,
)
async def test_source_connection(
    source_id: int,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
    tenant_id: str | None = Query(None),
):
    """
    Test connectivity to a source via the production sync path.

    Uses ``fetch_item_refs(limit=1)`` to exercise the same HTTP path
    as a real sync.  Zero side effects — read-only.

    🔒 Admin-only, tenant-scoped.
    """
    scoped_tenant = _scope_tenant(admin, tenant_id)

    result = await _get_inv_svc().test_connection(
        db, tenant_id=scoped_tenant, source_id=source_id
    )
    if result is None:
        raise HTTPException(status_code=404, detail="Source not found")
    return result


@router.get(
    "/sources/{source_id}/preview-refs",
    response_model=SourcePreviewRefsResponse,
)
async def preview_source_refs(
    source_id: int,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
    tenant_id: str | None = Query(None),
    limit: int = Query(10, ge=1, le=100),
):
    """
    Preview item refs from a source.  Zero side effects.

    🔒 Admin-only, tenant-scoped.
    """
    scoped_tenant = _scope_tenant(admin, tenant_id)

    result = await _get_inv_svc().preview_refs(
        db,
        tenant_id=scoped_tenant,
        source_id=source_id,
        limit=limit,
    )
    if result is None:
        raise HTTPException(status_code=404, detail="Source not found")
    return result


@router.get(
    "/sources/{source_id}/items/{external_id}/preview",
    response_model=SourceItemPreviewResponse,
)
async def preview_source_item(
    source_id: int,
    external_id: str,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
    tenant_id: str | None = Query(None),
):
    """
    Preview a single item with raw, canonical, and validation layers.

    Zero side effects — read-only investigation endpoint.

    🔒 Admin-only, tenant-scoped.
    """
    scoped_tenant = _scope_tenant(admin, tenant_id)

    result = await _get_inv_svc().preview_item_detail(
        db,
        tenant_id=scoped_tenant,
        source_id=source_id,
        external_id=external_id,
    )
    if result is None:
        raise HTTPException(status_code=404, detail="Source not found")
    return result


# ── Phase 2B: Single-Item Sync Endpoint ──────────────────────────────


@router.post(
    "/sources/{source_id}/sync-one",
    response_model=SyncOneResponse,
)
async def sync_one_item(
    source_id: int,
    body: SyncOneRequest,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
    tenant_id: str | None = Query(None),
):
    """
    Sync exactly one item by external_id from a source.

    Fetches, maps, resolves delta, upserts document, and updates the
    source link — scoped to exactly one item.

    Transaction safety:
      - ok=true  → auto-committed by get_db()
      - ok=false → explicit rollback (prevents flushed writes from committing)
      - exception → explicit rollback + re-raise

    🔒 Admin-only, tenant-scoped.
    """
    scoped_tenant = _scope_tenant(admin, tenant_id)

    try:
        result = await _get_action_svc().sync_one(
            db,
            tenant_id=scoped_tenant,
            source_id=source_id,
            external_id=body.external_id,
            force_reprocess=body.force_reprocess,
        )
    except Exception:
        await db.rollback()
        raise

    if result is None:
        # Source not found — nothing was written, rollback is a no-op.
        raise HTTPException(status_code=404, detail="Source not found")

    if not result.ok:
        # Soft failure path: any flushed writes must not be committed.
        await db.rollback()

    return result


# ── Phase 2C: Link Repair Endpoint ──────────────────────────────────


@router.post(
    "/sources/{source_id}/links/{link_id}/repair",
    response_model=SourceRepairResponse,
)
async def repair_source_link(
    source_id: int,
    link_id: int,
    body: SourceRepairRequest,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
    tenant_id: str | None = Query(None),
):
    """
    Repair / reprocess exactly one source link by link_id.

    Reads the link's ``external_id``, re-fetches the item from the source,
    maps it to canonical form, runs delta logic, and upserts only that one
    link.  Only the specified link is mutated — no sweep, no batch, no
    full sync.

    Outcomes (``action_taken``):
      - ``reactivated``     — link was missing, item returned
      - ``updated``         — content checksum changed (or link in error)
      - ``unchanged``       — content identical, no upsert needed
      - ``force_reprocessed`` — ``force_reprocess=true`` overrode SKIP
      - ``failed``          — item not found, mapping error, or upsert error

    🔒 Admin-only, tenant-scoped.
    """
    scoped_tenant = _scope_tenant(admin, tenant_id)

    try:
        result = await _repair_svc.repair_link(
            db,
            tenant_id=scoped_tenant,
            source_id=source_id,
            link_id=link_id,
            force_reprocess=body.force_reprocess,
        )
    except Exception:
        await db.rollback()
        raise

    if result is None:
        # Source not found — nothing was written, rollback is a no-op.
        raise HTTPException(status_code=404, detail="Source not found")

    if not result.ok:
        # Soft failure path: any flushed writes must not be committed.
        await db.rollback()

    return result
