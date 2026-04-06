"""
Admin Documents API (Phase 2A + 2B).

READ-ONLY admin endpoints for document management (Phase 2A).
ACTION endpoints for retry/reindex/resynthesize (Phase 2B).

Endpoints:
  GET  /api/v1/admin/documents          — list
  GET  /api/v1/admin/documents/{id}     — detail
  GET  /api/v1/admin/documents/{id}/open — preview
  GET  /api/v1/admin/documents/{id}/download — download
  GET  /api/v1/admin/documents/{id}/history — timeline
  POST /api/v1/admin/documents/{id}/retry-ingest — retry ingest
  POST /api/v1/admin/documents/{id}/reindex — reindex vectors
  POST /api/v1/admin/documents/{id}/resynthesize — regenerate synthesis

RBAC:
  system_admin  → can view/act all tenants
  tenant_admin  → forced to own tenant
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import PlainTextResponse, Response
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth_deps import require_admin
from app.core.rbac import is_system_admin
from app.db.models.user import User
from app.db.session import get_db
from app.services.document_admin_service import DocumentAdminService
from app.services.document_event_emitter import (
    DOCUMENT_DOWNLOADED,
    DOCUMENT_OPENED,
    emit_document_event,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/admin",
    tags=["admin-documents"],
)

_svc = DocumentAdminService()


# ── Schemas ───────────────────────────────────────────────────────────


class DocumentListItem(BaseModel):
    id: int
    tenant_id: str
    title: str | None
    source: str
    external_id: str
    status: str
    representation_type: str
    parent_document_id: int | None
    created_at: datetime | None
    updated_at: datetime | None
    content_length: int
    content_type: str | None


class DocumentListResponse(BaseModel):
    items: list[DocumentListItem]
    total: int
    page: int
    page_size: int


class DocumentDetailResponse(BaseModel):
    document: dict[str, Any]
    metadata: dict[str, Any]
    content_stats: dict[str, Any]
    related: dict[str, Any]
    history_summary: list[dict[str, Any]]


class OpenResponse(BaseModel):
    document_id: int
    view: str
    title: str | None
    mime_type: str
    content: str
    truncated: bool
    full_length: int
    shown_length: int


class HistoryItem(BaseModel):
    id: str
    event_type: str
    from_status: str | None
    to_status: str | None
    actor_user_id: int | None
    request_id: str | None
    message: str | None
    metadata_json: dict[str, Any]
    created_at: datetime | None


class HistoryResponse(BaseModel):
    items: list[HistoryItem]


class ActionResponse(BaseModel):
    """Response for document action endpoints (Phase 2B)."""
    document_id: int
    action: str
    accepted: bool
    status: str
    message: str
    request_id: str


# ── Helpers ───────────────────────────────────────────────────────────


def _scope_tenant(admin: User, tenant_id_param: str | None) -> str:
    """Resolve effective tenant_id based on RBAC."""
    if is_system_admin(admin.role):
        return tenant_id_param or admin.tenant_id
    return admin.tenant_id


# ── Read Endpoints (Phase 2A) ────────────────────────────────────────


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
    q: str | None = Query(None, max_length=200),
    status: str | None = Query(None),
    representation_type: str | None = Query(None),
    source: str | None = Query(None),
    tenant_id: str | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
):
    """
    List documents for admin.

    🔒 Admin-only.
      - system_admin: can filter by tenant_id
      - tenant_admin: forced to own tenant
    """
    scoped_tenant = _scope_tenant(admin, tenant_id)

    items, total = await _svc.get_list(
        db,
        tenant_id=scoped_tenant,
        q=q,
        status=status,
        representation_type=representation_type,
        source=source,
        page=page,
        page_size=page_size,
    )

    return DocumentListResponse(
        items=[DocumentListItem(**item) for item in items],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/documents/{document_id}", response_model=DocumentDetailResponse)
async def get_document_detail(
    document_id: int,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
    tenant_id: str | None = Query(None),
):
    """
    Get document detail.

    🔒 Admin-only, tenant-scoped.
    """
    scoped_tenant = _scope_tenant(admin, tenant_id)

    detail = await _svc.get_detail(
        db, document_id=document_id, tenant_id=scoped_tenant,
    )
    if not detail:
        raise HTTPException(status_code=404, detail="Document not found")

    return DocumentDetailResponse(**detail)


@router.get("/documents/{document_id}/open", response_model=OpenResponse)
async def open_document(
    document_id: int,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
    view: str = Query("cleaned", pattern="^(raw|cleaned|synthesized)$"),
    tenant_id: str | None = Query(None),
):
    """
    Preview document content.

    view: raw | cleaned | synthesized
    Content is truncated at 20,000 chars for safety.

    🔒 Admin-only, tenant-scoped.
    """
    scoped_tenant = _scope_tenant(admin, tenant_id)

    payload = await _svc.get_open_payload(
        db,
        document_id=document_id,
        tenant_id=scoped_tenant,
        view=view,
    )
    if not payload:
        raise HTTPException(status_code=404, detail="Document not found")

    # Emit open event (fail-open)
    try:
        await emit_document_event(
            db,
            tenant_id=scoped_tenant,
            document_id=document_id,
            event_type=DOCUMENT_OPENED,
            actor_user_id=admin.id,
            metadata_json={"view": view},
        )
    except Exception:
        pass  # fail-open

    return OpenResponse(
        document_id=payload.document_id,
        view=payload.view,
        title=payload.title,
        mime_type=payload.mime_type,
        content=payload.content,
        truncated=payload.truncated,
        full_length=payload.full_length,
        shown_length=payload.shown_length,
    )


@router.get("/documents/{document_id}/download")
async def download_document(
    document_id: int,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
    view: str = Query("cleaned", pattern="^(raw|cleaned|synthesized)$"),
    format: str = Query("txt", pattern="^(txt|md|json)$"),
    tenant_id: str | None = Query(None),
):
    """
    Download document representation.

    view: raw | cleaned | synthesized
    format: txt | md | json

    🔒 Admin-only, tenant-scoped.
    """
    scoped_tenant = _scope_tenant(admin, tenant_id)

    payload = await _svc.build_download_payload(
        db,
        document_id=document_id,
        tenant_id=scoped_tenant,
        view=view,
        format=format,
    )
    if not payload:
        raise HTTPException(status_code=404, detail="Document not found")

    # Emit download event (fail-open)
    try:
        await emit_document_event(
            db,
            tenant_id=scoped_tenant,
            document_id=document_id,
            event_type=DOCUMENT_DOWNLOADED,
            actor_user_id=admin.id,
            metadata_json={"view": view, "format": format},
        )
    except Exception:
        pass  # fail-open

    return Response(
        content=payload.content.encode("utf-8"),
        media_type=payload.mime_type,
        headers={
            "Content-Disposition": f'attachment; filename="{payload.filename}"',
        },
    )


@router.get("/documents/{document_id}/history", response_model=HistoryResponse)
async def get_document_history(
    document_id: int,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
    tenant_id: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
):
    """
    Get document event history timeline.

    Newest events first.

    🔒 Admin-only, tenant-scoped.
    """
    scoped_tenant = _scope_tenant(admin, tenant_id)

    events = await _svc.get_history(
        db,
        document_id=document_id,
        tenant_id=scoped_tenant,
        limit=limit,
    )
    if events is None:
        raise HTTPException(status_code=404, detail="Document not found")

    return HistoryResponse(
        items=[HistoryItem(**e) for e in events],
    )


# ── Action Endpoints (Phase 2B) ──────────────────────────────────────


@router.post(
    "/documents/{document_id}/retry-ingest",
    response_model=ActionResponse,
)
async def retry_ingest(
    document_id: int,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
    tenant_id: str | None = Query(None),
):
    """
    Retry document ingest from stored content.

    Re-runs the full pipeline (clean → chunk → embed → index) using the
    stored content_raw. Does NOT re-download the original file.

    Preconditions: status ∈ {error, uploaded, chunked, indexed}

    🔒 Admin-only, tenant-scoped.
    """
    from app.services.document_action_service import DocumentActionService

    scoped_tenant = _scope_tenant(admin, tenant_id)
    action_svc = DocumentActionService()

    result = await action_svc.retry_ingest(
        db,
        document_id=document_id,
        tenant_id=scoped_tenant,
        actor_user_id=admin.id,
    )

    if not result.accepted and result.status == "not_found":
        raise HTTPException(status_code=404, detail=result.message)

    return ActionResponse(
        document_id=result.document_id,
        action=result.action,
        accepted=result.accepted,
        status=result.status,
        message=result.message,
        request_id=result.request_id,
    )


@router.post(
    "/documents/{document_id}/reindex",
    response_model=ActionResponse,
)
async def reindex_document(
    document_id: int,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
    tenant_id: str | None = Query(None),
):
    """
    Reindex document vectors.

    Re-chunks the stored content_text, re-embeds, and replaces vectors.
    Does NOT re-process from raw content.

    Preconditions: status ∈ {chunked, indexed, ready}

    🔒 Admin-only, tenant-scoped.
    """
    from app.services.document_action_service import DocumentActionService

    scoped_tenant = _scope_tenant(admin, tenant_id)
    action_svc = DocumentActionService()

    result = await action_svc.reindex(
        db,
        document_id=document_id,
        tenant_id=scoped_tenant,
        actor_user_id=admin.id,
    )

    if not result.accepted and result.status == "not_found":
        raise HTTPException(status_code=404, detail=result.message)

    return ActionResponse(
        document_id=result.document_id,
        action=result.action,
        accepted=result.accepted,
        status=result.status,
        message=result.message,
        request_id=result.request_id,
    )


@router.post(
    "/documents/{document_id}/resynthesize",
    response_model=ActionResponse,
)
async def resynthesize_document(
    document_id: int,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
    tenant_id: str | None = Query(None),
):
    """
    Regenerate synthesized child representation.

    Calls the synthesis engine to create/update a synthesized child document
    from the original. Only works for original documents.

    Preconditions: representation_type = 'original', has content

    🔒 Admin-only, tenant-scoped.
    """
    from app.services.document_action_service import DocumentActionService

    scoped_tenant = _scope_tenant(admin, tenant_id)
    action_svc = DocumentActionService()

    result = await action_svc.resynthesize(
        db,
        document_id=document_id,
        tenant_id=scoped_tenant,
        actor_user_id=admin.id,
    )

    if not result.accepted and result.status == "not_found":
        raise HTTPException(status_code=404, detail=result.message)

    return ActionResponse(
        document_id=result.document_id,
        action=result.action,
        accepted=result.accepted,
        status=result.status,
        message=result.message,
        request_id=result.request_id,
    )

