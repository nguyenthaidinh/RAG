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
from app.schemas.document_reference import IngestReferenceRequest, IngestReferenceResponse
from app.core.config import settings
from app.schemas.ingest_mode import IngestMode
from app.services.ingest_strategy.factory import IngestStrategyFactory
from app.services.file_service_sync import FileServiceSyncClient

from app.services.document_extract import (
    extract_text,
    UnsupportedFileType,
    FileTooLarge,
    ExtractedTextTooLarge,
    ExtractLimits,
)
from app.services.remote_file_fetcher import (
    RemoteFileFetcher,
    RemoteFetchError,
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
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(e),
        )

    except IntegrityError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Conflict: document constraint violation",
        )

    except Exception:
        logger.exception(
            "documents.upsert_failed tenant_id=%s user_id=%s request_id=%s",
            tenant_id,
            getattr(current_user, "id", None),
            getattr(http_request.state, "request_id", None),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.post("/upload", response_model=UpsertDocumentResponse)
async def upload_document(
    http_request: FastAPIRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    file: UploadFile = File(...),
    original_name: str | None = Form(None),
    source: str = Form("core-platform"),
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
            raise HTTPException(
                status_code=422,
                detail="metadata_json không hợp lệ (phải là JSON object).",
            )

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
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(e),
        )
    except IntegrityError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Conflict: document constraint violation",
        )
    except Exception:
        logger.exception(
            "documents.upload_failed tenant_id=%s user_id=%s request_id=%s",
            tenant_id,
            getattr(current_user, "id", None),
            getattr(http_request.state, "request_id", None),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


_fetcher = RemoteFileFetcher()


@router.post("/ingest-reference", response_model=IngestReferenceResponse)
async def ingest_reference(
    body: IngestReferenceRequest,
    http_request: FastAPIRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    tenant_id = current_user.tenant_id
    request_id = getattr(http_request.state, "request_id", None)

    logger.info(
        "documents.ingest_reference.start tenant_id=%s source=%s external_id=%s provider=%s request_id=%s",
        tenant_id,
        body.source,
        body.external_id,
        body.reference.provider,
        request_id,
    )

    meta = body.metadata or {}

    try:
        fetch_result = await _fetcher.fetch(
            body.reference.temporary_url,
            provider=body.reference.provider,
        )
    except RemoteFetchError as exc:
        logger.warning(
            "documents.ingest_reference.fetch_failed tenant_id=%s external_id=%s error=%s",
            tenant_id,
            body.external_id,
            str(exc),
        )
        raise HTTPException(status_code=exc.status_code, detail=str(exc))

    data = fetch_result.content

    filename_for_extract = (
        fetch_result.inferred_filename
        or meta.get("original_name")
        or body.reference.original_name
        or "document"
    )

    final_title = (
        body.title
        or meta.get("original_name")
        or body.reference.original_name
        or fetch_result.inferred_filename
        or "document"
    )

    content_type = (
        fetch_result.content_type
        or meta.get("mime_type")
        or body.reference.mime_type
    )

    try:
        text = extract_text(
            filename_for_extract,
            content_type,
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
            detail="Không trích xuất được nội dung từ file tải về (file rỗng hoặc định dạng không đọc được).",
        )

    resolved_mode = _resolve_ingest_mode(getattr(body, "ingest_mode", None))
    strategy = IngestStrategyFactory.create(resolved_mode)

    resolved_file_id = meta.get("file_id") or meta.get("file_service_file_id")

    raw_metadata: dict[str, Any] = dict(meta)
    raw_metadata.update(
        {
            "provider": body.reference.provider,
            "content_type": content_type,
            "size_bytes": len(data),
            "inferred_filename": filename_for_extract,
            "file_id": resolved_file_id,
            "temporary_url_source": "redacted",
            "source_ref": {
                "provider": body.reference.provider,
                "bucket": body.reference.bucket,
                "key": body.reference.key,
                "original_name": body.reference.original_name,
                "mime_type": body.reference.mime_type,
                "size_bytes": body.reference.size_bytes,
                "file_id": resolved_file_id,
            },
        }
    )

    # Nếu bạn muốn external_id của ingest-reference cũng tách theo mode như /upload:
    # effective_external_id = f"{body.external_id}:{resolved_mode.value}"
    effective_external_id = body.external_id

    final_metadata = strategy.build_metadata(
        title=final_title,
        text=text,
        file_name=filename_for_extract,
        original_name=meta.get("original_name") or body.reference.original_name,
        content_type=content_type,
        size_bytes=len(data),
        ingest_via="reference",
        raw_metadata=raw_metadata,
    )

    service = DocumentService()

    try:
        doc, action, changed = await service.upsert(
            db=db,
            tenant_id=tenant_id,
            source=body.source,
            external_id=effective_external_id,
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
                request_id=request_id,
                document_id=doc.id,
                action=action,
                source=body.source,
                external_id=effective_external_id,
            )
        except Exception:
            logger.warning("audit.document_uploaded_failed", exc_info=True)

        await maybe_synthesize_child(
            db,
            original_doc=doc,
            original_text=doc.content_text or text,
            tenant_id=tenant_id,
        )

        try:
            file_id_raw = resolved_file_id
            if body.reference.provider == "file-service" and file_id_raw is not None:
                sync_client = FileServiceSyncClient()
                await sync_client.sync_document_metadata(
                    file_id=int(file_id_raw),
                    temporary_url=body.reference.temporary_url,
                    metadata_json=final_metadata,
                    ai_document_id=doc.id,
                    semantic_status="ready",
                )
        except Exception:
            logger.warning(
                "documents.ingest_reference.file_service_sync_failed tenant_id=%s external_id=%s",
                tenant_id,
                body.external_id,
                exc_info=True,
            )

        logger.info(
            "documents.ingest_reference.done tenant_id=%s doc_id=%s action=%s source=%s external_id=%s request_id=%s",
            tenant_id,
            doc.id,
            action,
            body.source,
            effective_external_id,
            request_id,
        )

        return IngestReferenceResponse(
            status="ok",
            action=action,
            document_id=doc.id,
            source=body.source,
            external_id=effective_external_id,
            changed=changed,
        )

    except QuotaExceededError as e:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(e),
        )

    except IntegrityError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Conflict: document constraint violation",
        )

    except Exception:
        logger.exception(
            "documents.ingest_reference.failed tenant_id=%s source=%s external_id=%s request_id=%s",
            tenant_id,
            body.source,
            body.external_id,
            request_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )
    
# import logging
# import hashlib
# import json
# from typing import Any

# from fastapi import (
#     APIRouter,
#     Depends,
#     HTTPException,
#     Request as FastAPIRequest,
#     status,
#     UploadFile,
#     File,
#     Form,
# )
# from sqlalchemy.ext.asyncio import AsyncSession
# from sqlalchemy.exc import IntegrityError

# from app.db.session import get_db
# from app.core.auth_deps import get_current_user
# from app.db.models.user import User
# from app.services.document_service import DocumentService
# from app.services.exceptions import QuotaExceededError
# from app.schemas.document import UpsertDocumentRequest, UpsertDocumentResponse
# from app.schemas.document_reference import IngestReferenceRequest, IngestReferenceResponse
# from app.core.config import settings
# from app.schemas.ingest_mode import IngestMode
# from app.services.ingest_strategy.factory import IngestStrategyFactory
# from app.services.file_service_sync import FileServiceSyncClient

# from app.services.document_extract import (
#     extract_text,
#     UnsupportedFileType,
#     FileTooLarge,
#     ExtractedTextTooLarge,
#     ExtractLimits,
# )
# from app.services.remote_file_fetcher import (
#     RemoteFileFetcher,
#     RemoteFetchError,
# )
# from app.services.synthesis_orchestrator import maybe_synthesize_child

# logger = logging.getLogger(__name__)

# router = APIRouter(prefix="/api/v1/documents", tags=["documents"])

# def _resolve_ingest_mode(requested_mode: IngestMode | None) -> IngestMode:
#     default_mode_raw = getattr(settings, "DOCUMENT_INGEST_MODE", "legacy")
#     try:
#         default_mode = IngestMode(default_mode_raw)
#     except Exception:
#         default_mode = IngestMode.LEGACY

#     allow_override = bool(getattr(settings, "DOCUMENT_INGEST_ALLOW_OVERRIDE", True))
#     if requested_mode and allow_override:
#         return requested_mode

#     return default_mode


# @router.post("/upsert", response_model=UpsertDocumentResponse)
# async def upsert_document(
#     request: UpsertDocumentRequest,
#     http_request: FastAPIRequest,
#     db: AsyncSession = Depends(get_db),
#     current_user: User = Depends(get_current_user),
# ):
#     tenant_id = current_user.tenant_id
#     service = DocumentService()

#     resolved_mode = _resolve_ingest_mode(request.ingest_mode)
#     strategy = IngestStrategyFactory.create(resolved_mode)

#     final_metadata = strategy.build_metadata(
#         title=request.title,
#         text=request.content,
#         file_name=None,
#         original_name=None,
#         content_type="text/plain",
#         size_bytes=len(request.content.encode("utf-8")),
#         ingest_via="upsert",
#         raw_metadata=request.metadata,
#     )

#     effective_external_id = f"{request.external_id}:{resolved_mode.value}"

#     try:
#         doc, action, changed = await service.upsert(
#             db=db,
#             tenant_id=tenant_id,
#             source=request.source,
#             # external_id=request.external_id,
#             # content=request.content,
#             # title=request.title,
#             # metadata=request.metadata,
#             external_id=effective_external_id,
#             content=request.content,
#             title=request.title,
#             metadata=final_metadata,
#         )

#         # ── Phase 6: emit DOCUMENT_UPLOADED audit (fail-open) ──
#         try:
#             from app.services.audit_service import get_audit_service

#             await get_audit_service().log_document_uploaded(
#                 db,
#                 tenant_id=tenant_id,
#                 user_id=current_user.id,
#                 request_id=getattr(http_request.state, "request_id", None),
#                 document_id=doc.id,
#                 action=action,
#                 source=request.source,
#                 # external_id=request.external_id,
#                 external_id=effective_external_id,
#             )
#         except Exception:
#             logger.warning("audit.document_uploaded_failed", exc_info=True)

#         return UpsertDocumentResponse(
#             status="ok",
#             action=action,
#             document_id=doc.id,
#             changed=changed,
#         )

#     except QuotaExceededError as e:
#         raise HTTPException(
#             status_code=status.HTTP_429_TOO_MANY_REQUESTS,
#             detail=str(e),
#         )

#     except IntegrityError:
#         raise HTTPException(
#             status_code=status.HTTP_409_CONFLICT,
#             detail="Conflict: document constraint violation",
#         )

#     except Exception:
#         logger.exception(
#             "documents.upsert_failed tenant_id=%s user_id=%s request_id=%s",
#             tenant_id,
#             getattr(current_user, "id", None),
#             getattr(http_request.state, "request_id", None),
#         )
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Internal server error",
#         )


# @router.post("/upload", response_model=UpsertDocumentResponse)
# async def upload_document(
#     http_request: FastAPIRequest,
#     db: AsyncSession = Depends(get_db),
#     current_user: User = Depends(get_current_user),
#     file: UploadFile = File(...),

#     # compat với core-platform (hay gửi original_name)
#     original_name: str | None = Form(None),

#     # chuẩn pro
#     source: str = Form("core-platform"),
#     external_id: str | None = Form(None),
#     title: str | None = Form(None),

#     # JSON string (object) optional
#     metadata_json: str | None = Form(None),
# ):
#     tenant_id = current_user.tenant_id
#     service = DocumentService()

#     data = await file.read()

#     # Parse metadata_json (must be object)
#     metadata: dict[str, Any] = {}
#     if metadata_json:
#         try:
#             parsed = json.loads(metadata_json)
#             if not isinstance(parsed, dict):
#                 raise ValueError("metadata_json must be an object")
#             metadata.update(parsed)
#         except Exception:
#             raise HTTPException(
#                 status_code=422,
#                 detail="metadata_json không hợp lệ (phải là JSON object).",
#             )

#     # Deterministic external_id if missing (idempotent)
#     if not external_id:
#         digest = hashlib.sha256(data).hexdigest()
#         external_id = f"{source}:{digest}"

#     final_title = title or original_name or file.filename

#     # Extract text (limit theo schema: output max 5MB utf-8 bytes)
#     try:
#         text = extract_text(
#             file.filename,
#             file.content_type,
#             data,
#             limits=ExtractLimits(
#                 max_bytes=15 * 1024 * 1024,      # input file limit
#                 max_chars=1_000_000,             # guard
#                 max_text_bytes=5 * 1024 * 1024,  # MUST match UpsertDocumentRequest validator
#             ),
#         )
#     except FileTooLarge as e:
#         raise HTTPException(status_code=413, detail=str(e))
#     except ExtractedTextTooLarge as e:
#         raise HTTPException(status_code=422, detail=str(e))
#     except UnsupportedFileType as e:
#         raise HTTPException(status_code=415, detail=str(e))

#     if not text:
#         raise HTTPException(
#             status_code=422,
#             detail="Không trích xuất được nội dung (file rỗng hoặc không đọc được).",
#         )

#     # Enrich metadata (KHÔNG log raw text/file bytes)
#     metadata.update(
#         {
#             "ingest_via": "upload",
#             "file_name": file.filename,
#             "original_name": original_name,
#             "content_type": file.content_type,
#             "size_bytes": len(data),
#         }
#     )

#     try:
#         doc, action, changed = await service.upsert(
#             db=db,
#             tenant_id=tenant_id,
#             source=source,
#             external_id=external_id,
#             content=text,
#             title=final_title,
#             metadata=metadata,
#         )

#         # ── Phase 6: emit DOCUMENT_UPLOADED audit (fail-open) ──
#         try:
#             from app.services.audit_service import get_audit_service

#             await get_audit_service().log_document_uploaded(
#                 db,
#                 tenant_id=tenant_id,
#                 user_id=current_user.id,
#                 request_id=getattr(http_request.state, "request_id", None),
#                 document_id=doc.id,
#                 action=action,
#                 source=source,
#                 external_id=external_id,
#             )
#         except Exception:
#             logger.warning("audit.document_uploaded_failed", exc_info=True)

#         # ── Phase 9.0: best-effort synthesis of child document (fail-open) ──
#         await maybe_synthesize_child(
#             db,
#             original_doc=doc,
#             original_text=doc.content_text or text,
#             tenant_id=tenant_id,
#         )

#         return UpsertDocumentResponse(
#             status="ok",
#             action=action,
#             document_id=doc.id,
#             changed=changed,
#         )

#     except QuotaExceededError as e:
#         raise HTTPException(
#             status_code=status.HTTP_429_TOO_MANY_REQUESTS,
#             detail=str(e),
#         )

#     except IntegrityError:
#         raise HTTPException(
#             status_code=status.HTTP_409_CONFLICT,
#             detail="Conflict: document constraint violation",
#         )

#     except Exception:
#         logger.exception(
#             "documents.upload_failed tenant_id=%s user_id=%s request_id=%s",
#             tenant_id,
#             getattr(current_user, "id", None),
#             getattr(http_request.state, "request_id", None),
#         )
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Internal server error",
#         )
