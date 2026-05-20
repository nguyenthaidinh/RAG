"""
CTDT Ingest Service — Tải file từ FileServer URL và ingest vào RAG pipeline.

Luồng:
    validate request → download file (streaming) → verify checksum
    → extract text → DocumentService.upsert()
    → clean → chunk → embed → index → ready
    → persist stats vào Document.meta["ctdt"]

Reuse hoàn toàn pipeline hiện có. Metadata nghiệp vụ CTĐT lưu trong
Document.meta (JSONB) dưới key "ctdt".

R1.1 hardening:
  - Streaming download with early size rejection
  - Host allowlist for file_url (SSRF protection)
  - Checksum verification (SHA-256)
  - Persist text_length/chunk_count/ingest_status into meta (no re-chunk on GET)
  - Scoped external_id by update_cycle_id
"""
from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.services.document_extract import (
    ExtractLimits,
    ExtractedTextTooLarge,
    FileTooLarge,
    TableLimits,
    UnsupportedFileType,
    extract_text_with_metadata,
)
from app.services.document_service import DocumentPipelineStats, DocumentService
from app.schemas.ingest_mode import IngestMode

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────

CTDT_SOURCE = "cn_ctdt"
CTDT_SOURCE_MODULE = "update_cycle"


# ── Error helpers ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class IngestError:
    """Structured error for CTDT ingest pipeline."""
    code: str
    message: str
    retryable: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
        }


@dataclass
class IngestResult:
    """Result from a successful ingest operation."""
    ai_document_id: int
    external_file_id: str
    ingest_status: str
    text_length: int
    chunk_count: int
    embedding_count: int
    indexed_count: int
    indexed: bool
    vector_index: str
    embedding_provider: str
    message: str


class IngestValidationError(Exception):
    """Raised when ingest request validation fails."""
    def __init__(self, error: IngestError):
        self.error = error
        super().__init__(error.message)


class IngestPipelineError(Exception):
    """Raised when the ingest pipeline fails after download."""
    def __init__(self, error: IngestError):
        self.error = error
        super().__init__(error.message)


# ── Host allowlist ────────────────────────────────────────────────────

# RFC 1918 / loopback prefixes for private IP detection
_PRIVATE_PREFIXES = (
    "127.", "10.", "192.168.", "0.",
)
_PRIVATE_RANGES_172 = range(16, 32)  # 172.16.0.0 – 172.31.255.255


def _is_private_ip(host: str) -> bool:
    """Check if host looks like a private/loopback IP address."""
    for prefix in _PRIVATE_PREFIXES:
        if host.startswith(prefix):
            return True
    if host.startswith("172."):
        parts = host.split(".")
        if len(parts) >= 2:
            try:
                second_octet = int(parts[1])
                if second_octet in _PRIVATE_RANGES_172:
                    return True
            except ValueError:
                pass
    if host == "::1" or host.startswith("[::1]"):
        return True
    return False


def _get_allowed_hosts() -> set[str]:
    """Parse allowed host list from config."""
    raw = settings.RAG_REMOTE_FILE_ALLOWED_HOSTS
    return {h.strip().lower() for h in raw.split(",") if h.strip()}


def validate_file_url_host(file_url: str) -> None:
    """
    Validate file_url host against allowlist.

    Rules:
    - If allowlist is empty (dev mode): allow all, but still block
      localhost/private IP unless explicitly in allowlist.
    - If allowlist is set: only allow hosts in the list.
    """
    try:
        parsed = urlparse(file_url)
        host = (parsed.hostname or "").lower()
    except Exception:
        raise IngestValidationError(IngestError(
            code="invalid_file_url",
            message="URL không hợp lệ",
            retryable=False,
        ))

    if not host:
        raise IngestValidationError(IngestError(
            code="invalid_file_url",
            message="URL thiếu hostname",
            retryable=False,
        ))

    allowed_hosts = _get_allowed_hosts()

    if allowed_hosts:
        # Strict mode: host must be in allowlist
        if host not in allowed_hosts:
            raise IngestValidationError(IngestError(
                code="invalid_file_url",
                message="Host của file_url không nằm trong danh sách cho phép",
                retryable=False,
            ))
    else:
        # Dev mode (empty allowlist): block private/loopback IPs
        is_localhost = host in ("localhost", "127.0.0.1", "::1", "[::1]")
        if is_localhost or _is_private_ip(host):
            raise IngestValidationError(IngestError(
                code="invalid_file_url",
                message="Không cho phép tải file từ địa chỉ nội bộ. "
                        "Cấu hình RAG_REMOTE_FILE_ALLOWED_HOSTS nếu cần.",
                retryable=False,
            ))


# ── MIME type validation ──────────────────────────────────────────────

def _get_allowed_mime_types() -> set[str]:
    raw = settings.RAG_ALLOWED_MIME_TYPES
    return {m.strip().lower() for m in raw.split(",") if m.strip()}


def validate_mime_type(mime_type: str) -> None:
    """Validate MIME type against allowed list."""
    normalized = mime_type.strip().lower().split(";", 1)[0].strip()
    allowed = _get_allowed_mime_types()
    if normalized not in allowed:
        raise IngestValidationError(IngestError(
            code="unsupported_mime_type",
            message=f"MIME type '{normalized}' không được hỗ trợ. Cho phép: {', '.join(sorted(allowed))}",
            retryable=False,
        ))


def validate_file_url(file_url: str) -> None:
    """Basic URL scheme validation."""
    if not file_url or not file_url.startswith(("http://", "https://")):
        raise IngestValidationError(IngestError(
            code="invalid_file_url",
            message="file_url phải là URL hợp lệ (http:// hoặc https://)",
            retryable=False,
        ))


# ── Checksum verification ────────────────────────────────────────────

def verify_checksum(data: bytes, expected_checksum: str) -> None:
    """
    Verify SHA-256 checksum of downloaded data.

    Raises IngestPipelineError on mismatch.
    """
    actual = hashlib.sha256(data).hexdigest().lower()
    expected = expected_checksum.strip().lower()

    if actual != expected:
        logger.warning(
            "ctdt_ingest.checksum_mismatch expected=%s actual=%s size=%d",
            expected[:16] + "...", actual[:16] + "...", len(data),
        )
        raise IngestPipelineError(IngestError(
            code="checksum_mismatch",
            message="Checksum file không khớp (SHA-256). File có thể bị lỗi khi truyền.",
            retryable=False,
        ))


# ── File download (streaming) ────────────────────────────────────────

async def download_remote_file(
    file_url: str,
    *,
    timeout_seconds: int | None = None,
    max_mb: int | None = None,
    user_agent: str | None = None,
) -> bytes:
    """
    Download file from remote URL with streaming, timeout and size limit.

    - Checks Content-Length header for early rejection
    - Streams body in chunks, accumulating and checking size
    - Never reads more than max_bytes into RAM

    Returns raw bytes. Raises IngestPipelineError on failure.
    """
    timeout = timeout_seconds or settings.RAG_REMOTE_FILE_TIMEOUT_SECONDS
    max_size_mb = max_mb or settings.RAG_REMOTE_FILE_MAX_MB
    max_bytes = max_size_mb * 1024 * 1024
    ua = user_agent or settings.RAG_DOWNLOAD_USER_AGENT

    t0 = time.monotonic()

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=15.0),
            follow_redirects=True,
            max_redirects=5,
        ) as client:
            async with client.stream(
                "GET",
                file_url,
                headers={"User-Agent": ua},
            ) as response:
                response.raise_for_status()

                # ── Early reject via Content-Length ────────────────
                content_length = response.headers.get("content-length")
                if content_length:
                    try:
                        declared_size = int(content_length)
                        if declared_size > max_bytes:
                            logger.warning(
                                "ctdt_ingest.content_length_too_large "
                                "declared=%d max=%d",
                                declared_size, max_bytes,
                            )
                            raise IngestPipelineError(IngestError(
                                code="file_too_large",
                                message=(
                                    f"File quá lớn: {declared_size} bytes "
                                    f"> {max_bytes} bytes ({max_size_mb}MB)"
                                ),
                                retryable=False,
                            ))
                    except ValueError:
                        pass  # Malformed Content-Length, continue streaming

                # ── Stream body with running size check ───────────
                chunks: list[bytes] = []
                received = 0
                async for chunk in response.aiter_bytes(chunk_size=64 * 1024):
                    received += len(chunk)
                    if received > max_bytes:
                        logger.warning(
                            "ctdt_ingest.stream_too_large received=%d max=%d",
                            received, max_bytes,
                        )
                        raise IngestPipelineError(IngestError(
                            code="file_too_large",
                            message=(
                                f"File quá lớn: đã nhận {received} bytes "
                                f"> {max_bytes} bytes ({max_size_mb}MB)"
                            ),
                            retryable=False,
                        ))
                    chunks.append(chunk)

    except IngestPipelineError:
        raise  # Re-raise our own errors

    except httpx.TimeoutException:
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        logger.warning(
            "ctdt_ingest.download_timeout url_length=%d timeout=%ds elapsed_ms=%d",
            len(file_url), timeout, elapsed_ms,
        )
        raise IngestPipelineError(IngestError(
            code="download_failed",
            message=f"Timeout khi tải file (>{timeout}s)",
            retryable=True,
        ))

    except httpx.HTTPStatusError as exc:
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        logger.warning(
            "ctdt_ingest.download_http_error status=%d url_length=%d elapsed_ms=%d",
            exc.response.status_code, len(file_url), elapsed_ms,
        )
        raise IngestPipelineError(IngestError(
            code="download_failed",
            message=f"FileServer trả HTTP {exc.response.status_code}",
            retryable=exc.response.status_code >= 500,
        ))

    except Exception as exc:
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        logger.warning(
            "ctdt_ingest.download_error error_type=%s url_length=%d elapsed_ms=%d",
            type(exc).__name__, len(file_url), elapsed_ms,
        )
        raise IngestPipelineError(IngestError(
            code="download_failed",
            message="Không thể tải file từ FileServer",
            retryable=True,
        ))

    data = b"".join(chunks)
    elapsed_ms = int((time.monotonic() - t0) * 1000)

    logger.info(
        "ctdt_ingest.downloaded size_bytes=%d elapsed_ms=%d",
        len(data), elapsed_ms,
    )

    return data


# ── Metadata builder ─────────────────────────────────────────────────

def build_ctdt_metadata(
    *,
    external_file_id: str,
    update_cycle_id: str,
    program_id: str | None,
    program_code: str | None,
    program_name: str | None,
    document_role: str,
    uploaded_by: str | None,
    checksum: str | None,
    filename: str,
    mime_type: str,
    file_size_bytes: int,
    ingest_mode: str,
) -> dict[str, Any]:
    """Build Document.meta JSONB with CTĐT business metadata."""
    return {
        "system": {
            "pipeline_mode": ingest_mode,
            "pipeline_version": f"{ingest_mode}_v1",
            "ingest_via": "ctdt_remote_url",
            "file_name": filename,
            "original_name": filename,
            "content_type": mime_type,
            "size_bytes": file_size_bytes,
            "force_reprocess_if_pipeline_stats_missing": True,
        },
        "ctdt": {
            "source_system": CTDT_SOURCE,
            "source_module": CTDT_SOURCE_MODULE,
            "external_file_id": external_file_id,
            "update_cycle_id": update_cycle_id,
            "program_id": program_id,
            "program_code": program_code,
            "program_name": program_name,
            "document_role": document_role,
            "uploaded_by": uploaded_by,
            "checksum": checksum,
        },
        "user_metadata": {},
    }


# ── Main ingest orchestrator ─────────────────────────────────────────

def _resolve_ingest_mode() -> IngestMode:
    """Resolve ingest mode from config (same logic as documents router)."""
    raw = getattr(settings, "DOCUMENT_INGEST_MODE", "legacy")
    try:
        return IngestMode(raw)
    except Exception:
        return IngestMode.LEGACY


async def ingest_from_url(
    db: AsyncSession,
    *,
    tenant_id: str,
    external_file_id: str,
    file_url: str,
    filename: str,
    mime_type: str,
    checksum: str | None,
    update_cycle_id: str,
    program_id: str | None,
    program_code: str | None,
    program_name: str | None,
    document_role: str,
    uploaded_by: str | None,
) -> IngestResult:
    """
    Full CTĐT ingest pipeline: validate → download → verify → extract → upsert.

    Reuses DocumentService.upsert() for the heavy lifting
    (clean → chunk → embed → index → ready).

    Returns IngestResult on success. Raises IngestValidationError or
    IngestPipelineError on failure.
    """
    t0 = time.monotonic()

    # ── Step 1: Validate ──────────────────────────────────────────
    validate_file_url(file_url)
    validate_file_url_host(file_url)
    validate_mime_type(mime_type)

    # ── Step 2: Download (streaming) ──────────────────────────────
    data = await download_remote_file(file_url)

    # ── Step 2.5: Verify checksum ─────────────────────────────────
    if checksum:
        verify_checksum(data, checksum)

    # ── Step 3: Extract text ──────────────────────────────────────
    try:
        text, extraction_meta = extract_text_with_metadata(
            filename,
            mime_type,
            data,
            limits=ExtractLimits(
                max_bytes=settings.RAG_REMOTE_FILE_MAX_MB * 1024 * 1024,
                max_chars=settings.RAG_EXTRACT_MAX_TEXT_CHARS,
                max_text_bytes=5 * 1024 * 1024,
            ),
            table_limits=TableLimits(
                max_rows=settings.RAG_EXTRACT_MAX_TABLE_ROWS,
                max_cols=settings.RAG_EXTRACT_MAX_TABLE_COLS,
                include_empty_cells=settings.RAG_EXTRACT_INCLUDE_EMPTY_CELLS,
            ),
        )
    except FileTooLarge as exc:
        raise IngestPipelineError(IngestError(
            code="file_too_large",
            message=str(exc),
            retryable=False,
        ))
    except ExtractedTextTooLarge as exc:
        raise IngestPipelineError(IngestError(
            code="extract_failed",
            message=str(exc),
            retryable=False,
        ))
    except UnsupportedFileType as exc:
        raise IngestPipelineError(IngestError(
            code="unsupported_mime_type",
            message=str(exc),
            retryable=False,
        ))
    except Exception as exc:
        logger.warning(
            "ctdt_ingest.extract_failed filename=%s error_type=%s",
            filename, type(exc).__name__,
        )
        raise IngestPipelineError(IngestError(
            code="extract_failed",
            message="Không trích xuất được nội dung từ file",
            retryable=False,
        ))

    if not text or not text.strip():
        raise IngestPipelineError(IngestError(
            code="extract_failed",
            message="File rỗng hoặc không trích xuất được nội dung text",
            retryable=False,
        ))

    # ── Step 4: Build metadata ────────────────────────────────────
    resolved_mode = _resolve_ingest_mode()

    metadata = build_ctdt_metadata(
        external_file_id=external_file_id,
        update_cycle_id=update_cycle_id,
        program_id=program_id,
        program_code=program_code,
        program_name=program_name,
        document_role=document_role,
        uploaded_by=uploaded_by,
        checksum=checksum,
        filename=filename,
        mime_type=mime_type,
        file_size_bytes=len(data),
        ingest_mode=resolved_mode.value,
    )

    # R2: persist extraction stats (table/sheet counts) into meta
    if extraction_meta:
        metadata["extraction"] = extraction_meta

    # Scoped external_id: prevents collision across update cycles
    effective_external_id = (
        f"ctdt:update-cycle:{update_cycle_id}"
        f":file:{external_file_id}:{resolved_mode.value}"
    )

    # ── Step 5: Upsert via DocumentService ────────────────────────
    # This runs the full pipeline: create/update → clean → chunk → embed → index → ready
    service = DocumentService()

    try:
        doc, action, changed, pipeline_stats = await service.upsert(
            db=db,
            tenant_id=tenant_id,
            source=CTDT_SOURCE,
            external_id=effective_external_id,
            content=text,
            title=filename,
            metadata=metadata,
        )
    except Exception as exc:
        logger.exception(
            "ctdt_ingest.upsert_failed external_file_id=%s error_type=%s",
            external_file_id, type(exc).__name__,
        )
        raise IngestPipelineError(IngestError(
            code="index_failed",
            message="Lỗi khi xử lý và lập chỉ mục tài liệu",
            retryable=True,
        ))

    # ── Step 6: Compute & persist result stats ────────────────────
    # DocumentService already ran clean/chunk/embed/index. Do not re-clean or
    # re-chunk here; persist the stats returned by the real pipeline.
    text_length = pipeline_stats.cleaned_text_length
    chunk_count = pipeline_stats.chunk_count
    ingest_status = _map_status(doc.status)

    # Persist stats into meta["ctdt"] so GET status reads them
    # without re-processing the document.
    _persist_ingest_stats(
        doc,
        text_length,
        chunk_count,
        ingest_status,
        pipeline_stats=pipeline_stats,
    )
    await db.flush()

    elapsed_ms = int((time.monotonic() - t0) * 1000)

    logger.info(
        "ctdt_ingest.completed ai_document_id=%d external_file_id=%s "
        "status=%s text_length=%d chunk_count=%d action=%s elapsed_ms=%d",
        doc.id, external_file_id, ingest_status,
        text_length, chunk_count, action, elapsed_ms,
    )

    return IngestResult(
        ai_document_id=doc.id,
        external_file_id=external_file_id,
        ingest_status=ingest_status,
        text_length=text_length,
        chunk_count=chunk_count,
        embedding_count=pipeline_stats.embedding_count,
        indexed_count=pipeline_stats.indexed_count,
        indexed=pipeline_stats.indexed,
        vector_index=pipeline_stats.vector_index,
        embedding_provider=pipeline_stats.embedding_provider,
        message="Document ingested successfully",
    )


# ── Helpers ───────────────────────────────────────────────────────────

def _map_status(db_status: str) -> str:
    """Map internal document status to CTDT-facing ingest status."""
    mapping = {
        "uploaded": "extracting",
        "pending": "extracting",
        "chunked": "chunking",
        "processing": "chunking",
        "indexed": "indexing",
        "ready": "ready",
        "error": "failed",
    }
    return mapping.get(db_status, db_status)


def _compute_chunk_count(doc: Any, text: str) -> int:
    """
    Compute chunk count by running the NLP pipeline.

    Deprecated compatibility helper. CTDT ingest no longer calls this because
    DocumentService returns the real chunk count from the pipeline run.
    """
    from app.nlp import get_chunker, get_cleaner

    try:
        cleaner = get_cleaner()
        chunker = get_chunker()
        cleaned = cleaner.clean(doc.content_text or text)
        chunks = chunker.chunk(
            cleaned,
            tenant_id=doc.tenant_id,
            document_id=doc.id,
            version_id=doc.checksum,
        )
        return len(chunks)
    except Exception:
        # Fallback: rough estimate
        max_tokens = settings.NLP_CHUNK_MAX_TOKENS
        # ~1.5 chars per token for Vietnamese text
        estimated = max(1, len(text) // (max_tokens * 2))
        return estimated


def _persist_ingest_stats(
    doc: Any,
    text_length: int,
    chunk_count: int,
    ingest_status: str,
    *,
    pipeline_stats: DocumentPipelineStats | dict[str, Any] | None = None,
) -> None:
    """
    Persist computed stats into Document.meta["ctdt"].

    This avoids re-computing chunk_count on every GET status request.
    SQLAlchemy detects the mutation via dict copy.
    """
    meta = dict(doc.meta or {})
    ctdt = dict(meta.get("ctdt", {}))

    ctdt["text_length"] = text_length
    ctdt["chunk_count"] = chunk_count
    ctdt["ingest_status"] = ingest_status

    pipeline = dict(meta.get("pipeline", {}))
    if pipeline_stats is None:
        pipeline.update({
            "cleaned_text_length": text_length,
            "chunk_count": chunk_count,
            "embedding_count": 0,
            "indexed_count": 0,
            "indexed": False,
            "vector_index": "null"
            if settings.VECTOR_INDEX.lower().strip() == "null"
            else settings.VECTOR_INDEX,
            "embedding_provider": settings.EMBEDDING_PROVIDER,
        })
    else:
        stats_dict = (
            pipeline_stats.to_dict()
            if hasattr(pipeline_stats, "to_dict")
            else dict(pipeline_stats)
        )
        pipeline.update(stats_dict)
        ctdt["embedding_count"] = stats_dict.get("embedding_count", 0)
        ctdt["indexed_count"] = stats_dict.get("indexed_count", 0)
        ctdt["indexed"] = bool(stats_dict.get("indexed", False))
        ctdt["vector_index"] = stats_dict.get("vector_index", "null")
        ctdt["embedding_provider"] = stats_dict.get("embedding_provider")

    if str(pipeline.get("vector_index", "")).lower() == "null":
        pipeline["indexed"] = False
        pipeline["embedding_count"] = 0
        pipeline["indexed_count"] = 0
        ctdt["indexed"] = False
        ctdt["embedding_count"] = 0
        ctdt["indexed_count"] = 0
        ctdt["vector_index"] = "null"

    pipeline["ingest_status"] = ingest_status

    meta["ctdt"] = ctdt
    meta["pipeline"] = pipeline
    doc.meta = meta


def get_document_ctdt_info(doc: Any) -> dict[str, Any]:
    """
    Extract CTĐT metadata from a Document for status endpoint response.

    Reads from Document.meta["ctdt"]. Stats (text_length, chunk_count,
    ingest_status) are read from persisted meta — no re-processing.
    """
    meta = doc.meta or {}
    ctdt = meta.get("ctdt", {})

    # Read persisted stats; fallback to simple len() for text_length
    text_length = ctdt.get("text_length")
    if text_length is None:
        text_length = len(doc.content_text or doc.content_raw or "")

    chunk_count = ctdt.get("chunk_count")
    if chunk_count is None:
        # Lightweight fallback: rough estimate, no re-chunking
        max_tokens = settings.NLP_CHUNK_MAX_TOKENS
        raw_len = len(doc.content_text or doc.content_raw or "")
        chunk_count = max(1, raw_len // (max_tokens * 2)) if raw_len else 0

    ingest_status = ctdt.get("ingest_status")
    if ingest_status is None:
        ingest_status = _map_status(doc.status)

    return {
        "ai_document_id": doc.id,
        "external_file_id": ctdt.get("external_file_id"),
        "filename": doc.title,
        "document_role": ctdt.get("document_role"),
        "update_cycle_id": ctdt.get("update_cycle_id"),
        "program_id": ctdt.get("program_id"),
        "program_code": ctdt.get("program_code"),
        "program_name": ctdt.get("program_name"),
        "ingest_status": ingest_status,
        "text_length": text_length,
        "chunk_count": chunk_count,
        "embedding_count": ctdt.get(
            "embedding_count",
            (meta.get("pipeline", {}) or {}).get("embedding_count"),
        ),
        "indexed_count": ctdt.get(
            "indexed_count",
            (meta.get("pipeline", {}) or {}).get("indexed_count"),
        ),
        "indexed": ctdt.get(
            "indexed",
            (meta.get("pipeline", {}) or {}).get("indexed"),
        ),
        "vector_index": ctdt.get(
            "vector_index",
            (meta.get("pipeline", {}) or {}).get("vector_index"),
        ),
        "embedding_provider": ctdt.get(
            "embedding_provider",
            (meta.get("pipeline", {}) or {}).get("embedding_provider"),
        ),
        "error_message": None if ingest_status != "failed" else "Document processing failed",
        "uploaded_by": ctdt.get("uploaded_by"),
        "source_system": ctdt.get("source_system"),
        "source_module": ctdt.get("source_module"),
        "mime_type": (meta.get("system", {}).get("content_type")),
        "checksum": ctdt.get("checksum"),
        "created_at": doc.created_at,
        "updated_at": doc.updated_at,
    }
