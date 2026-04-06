import logging
import hashlib
import json
from typing import Any

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request as FastAPIRequest,
    status,
    UploadFile,
    File,
    Form,
)
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError

from app.db.session import get_db
from app.core.auth_deps import get_current_user
from app.db.models.user import User
from app.services.document_service import DocumentService
from app.services.exceptions import QuotaExceededError
from app.schemas.document import UpsertDocumentRequest, UpsertDocumentResponse
from app.core.config import settings
from app.schemas.ingest_mode import IngestMode
from app.services.ingest_strategy.factory import IngestStrategyFactory

from app.services.document_extract import (
    extract_text,
    UnsupportedFileType,
    FileTooLarge,
    ExtractedTextTooLarge,
    ExtractLimits,
)
from app.services.synthesis_orchestrator import maybe_synthesize_child

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


def _resolve_ingest_mode(requested_mode: IngestMode | None) -> IngestMode:
    default_mode_raw = getattr(settings, "DOCUMENT_INGEST_MODE", "legacy")
    try:
        default_mode = IngestMode(default_mode_raw)
    except Exception:
        default_mode = IngestMode.LEGACY

    allow_override = bool(getattr(settings, "DOCUMENT_INGEST_ALLOW_OVERRIDE", True))
    if requested_mode and allow_override:
        return requested_mode

    return default_mode


@router.post("/upsert", response_model=UpsertDocumentResponse)
async def upsert_document(
    request: UpsertDocumentRequest,
    http_request: FastAPIRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    tenant_id = current_user.tenant_id
    service = DocumentService()

    resolved_mode = _resolve_ingest_mode(request.ingest_mode)
    strategy = IngestStrategyFactory.create(resolved_mode)

    final_metadata = strategy.build_metadata(
        title=request.title,
        text=request.content,
        file_name=None,
        original_name=None,
        content_type="text/plain",
        size_bytes=len(request.content.encode("utf-8")),
        ingest_via="upsert",
        raw_metadata=request.metadata,
    )

    effective_external_id = f"{request.external_id}:{resolved_mode.value}"

    try:
        doc, action, changed = await service.upsert(
            db=db,
            tenant_id=tenant_id,
            source=request.source,
            external_id=effective_external_id,
            content=request.content,
            title=request.title,
            metadata=final_metadata,
        )

        try:
            from app.services.audit_service import get_audit_service
            await get_audit_service().log_document_uploaded(
                db,
                tenant_id=tenant_id,
                user_id=current_user.id,
                request_id=getattr(http_request.state, "request_id", None),
                document_id=doc.id,
                action=action,
                source=request.source,
                external_id=effective_external_id,
            )
        except Exception:
            logger.warning("audit.document_uploaded_failed", exc_info=True)

        return UpsertDocumentResponse(
            status="ok",
            action=action,
            document_id=doc.id,
            changed=changed,
        )

    except QuotaExceededError as e:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(e))
    except IntegrityError:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Conflict: document constraint violation")
    except Exception:
        logger.exception(
            "documents.upsert_failed tenant_id=%s user_id=%s request_id=%s",
            tenant_id, getattr(current_user, "id", None),
            getattr(http_request.state, "request_id", None),
        )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@router.post("/upload", response_model=UpsertDocumentResponse)
async def upload_document(
    http_request: FastAPIRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    file: UploadFile = File(...),
    original_name: str | None = Form(None),
    source: str = Form("moodle"),
    external_id: str | None = Form(None),
    title: str | None = Form(None),
    metadata_json: str | None = Form(None),
    ingest_mode: IngestMode | None = Form(None),
):
    tenant_id = current_user.tenant_id
    service = DocumentService()

    data = await file.read()

    metadata: dict[str, Any] = {}
    if metadata_json:
        try:
            parsed = json.loads(metadata_json)
            if not isinstance(parsed, dict):
                raise ValueError("metadata_json must be an object")
            metadata.update(parsed)
        except Exception:
            raise HTTPException(status_code=422, detail="metadata_json không hợp lệ (phải là JSON object).")

    resolved_mode = _resolve_ingest_mode(ingest_mode)
    strategy = IngestStrategyFactory.create(resolved_mode)

    if not external_id:
        digest = hashlib.sha256(data).hexdigest()
        external_id = f"{source}:{digest}"

    external_id = f"{external_id}:{resolved_mode.value}"
    final_title = title or original_name or file.filename

    try:
        text = extract_text(
            file.filename,
            file.content_type,
            data,
            limits=ExtractLimits(
                max_bytes=15 * 1024 * 1024,
                max_chars=1_000_000,
                max_text_bytes=5 * 1024 * 1024,
            ),
        )
    except FileTooLarge as e:
        raise HTTPException(status_code=413, detail=str(e))
    except ExtractedTextTooLarge as e:
        raise HTTPException(status_code=422, detail=str(e))
    except UnsupportedFileType as e:
        raise HTTPException(status_code=415, detail=str(e))

    if not text:
        raise HTTPException(
            status_code=422,
            detail="Không trích xuất được nội dung (file rỗng hoặc không đọc được).",
        )

    final_metadata = strategy.build_metadata(
        title=final_title,
        text=text,
        file_name=file.filename,
        original_name=original_name,
        content_type=file.content_type,
        size_bytes=len(data),
        ingest_via="upload",
        raw_metadata=metadata,
    )

    try:
        doc, action, changed = await service.upsert(
            db=db,
            tenant_id=tenant_id,
            source=source,
            external_id=external_id,
            content=text,
            title=final_title,
            metadata=final_metadata,
        )

        try:
            from app.services.audit_service import get_audit_service
            await get_audit_service().log_document_uploaded(
                db,
                tenant_id=tenant_id,
                user_id=current_user.id,
                request_id=getattr(http_request.state, "request_id", None),
                document_id=doc.id,
                action=action,
                source=source,
                external_id=external_id,
            )
        except Exception:
            logger.warning("audit.document_uploaded_failed", exc_info=True)

        await maybe_synthesize_child(
            db,
            original_doc=doc,
            original_text=doc.content_text or text,
            tenant_id=tenant_id,
        )

        return UpsertDocumentResponse(
            status="ok",
            action=action,
            document_id=doc.id,
            changed=changed,
        )

    except QuotaExceededError as e:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(e))
    except IntegrityError:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Conflict: document constraint violation")
    except Exception:
        logger.exception(
            "documents.upload_failed tenant_id=%s user_id=%s request_id=%s",
            tenant_id, getattr(current_user, "id", None),
            getattr(http_request.state, "request_id", None),
        )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")
