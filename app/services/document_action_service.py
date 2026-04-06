"""
Document action service (Phase 2B).

Mutation orchestration layer for admin document actions:
  - retry_ingest:    re-run full pipeline from stored content_raw
  - reindex:         re-chunk + re-embed + re-index from stored content_text
  - resynthesize:    regenerate synthesized child from original doc

Design rules:
  - Tenant-scoped — all actions require tenant_id
  - RBAC enforced at API layer, not here
  - Every action emits requested / started / succeeded / failed events
  - Fail-safe — action errors never corrupt the original document
  - Uses existing DocumentService / SynthesisOrchestrator boundaries
  - No raw content in event metadata
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.document import Document
from app.repos.document_admin_repo import DocumentAdminRepo
from app.services.document_event_emitter import emit_document_event
from app.services.document_lifecycle import (
    CHUNKED,
    ERROR,
    INDEXED,
    READY,
    UPLOADED,
    validate_transition,
)

logger = logging.getLogger(__name__)

# ── Event type constants ──────────────────────────────────────────────

RETRY_INGEST_REQUESTED = "document.retry_ingest_requested"
RETRY_INGEST_STARTED = "document.retry_ingest_started"
RETRY_INGEST_SUCCEEDED = "document.retry_ingest_succeeded"
RETRY_INGEST_FAILED = "document.retry_ingest_failed"

REINDEX_REQUESTED = "document.reindex_requested"
REINDEX_STARTED = "document.reindex_started"
REINDEX_SUCCEEDED = "document.reindex_succeeded"
REINDEX_FAILED = "document.reindex_failed"

RESYNTHESIZE_REQUESTED = "document.resynthesize_requested"
RESYNTHESIZE_STARTED = "document.resynthesize_started"
RESYNTHESIZE_SUCCEEDED = "document.resynthesize_succeeded"
RESYNTHESIZE_FAILED = "document.resynthesize_failed"

# States that allow each action
RETRY_INGEST_ALLOWED = frozenset({ERROR, UPLOADED, CHUNKED, INDEXED})
REINDEX_ALLOWED = frozenset({CHUNKED, INDEXED, READY})
RESYNTHESIZE_ALLOWED_TYPES = frozenset({"original"})


@dataclass(frozen=True)
class ActionResult:
    """Result of a document action."""

    document_id: int
    action: str
    accepted: bool
    status: str
    message: str
    request_id: str


class DocumentActionService:
    """
    Document admin action orchestration.

    Each action:
      1. Validates preconditions
      2. Emits requested event
      3. Runs the action
      4. Emits succeeded / failed event
    """

    def __init__(self, repo: DocumentAdminRepo | None = None) -> None:
        self._repo = repo or DocumentAdminRepo()

    # ── Retry Ingest ──────────────────────────────────────────────────

    async def retry_ingest(
        self,
        db: AsyncSession,
        *,
        document_id: int,
        tenant_id: str,
        actor_user_id: int | None = None,
    ) -> ActionResult:
        """
        Re-run ingest pipeline from stored content_raw.

        Preconditions:
          - Document exists in tenant
          - Status in RETRY_INGEST_ALLOWED
          - content_raw is not empty
        """
        request_id = str(uuid.uuid4())[:12]
        action = "retry_ingest"

        doc = await self._repo.get_document_by_id(
            db, document_id=document_id, tenant_id=tenant_id,
        )
        if not doc:
            return ActionResult(
                document_id=document_id, action=action,
                accepted=False, status="not_found",
                message="Document not found", request_id=request_id,
            )

        # Validate preconditions
        if doc.status not in RETRY_INGEST_ALLOWED:
            return ActionResult(
                document_id=document_id, action=action,
                accepted=False, status=doc.status,
                message=f"Retry ingest not allowed in status '{doc.status}'. "
                        f"Allowed: {sorted(RETRY_INGEST_ALLOWED)}",
                request_id=request_id,
            )

        if not doc.content_raw or not doc.content_raw.strip():
            return ActionResult(
                document_id=document_id, action=action,
                accepted=False, status=doc.status,
                message="No stored content available for retry",
                request_id=request_id,
            )

        # Emit requested
        await emit_document_event(
            db, tenant_id=tenant_id, document_id=document_id,
            event_type=RETRY_INGEST_REQUESTED,
            actor_user_id=actor_user_id,
            request_id=request_id,
            message="Retry ingest from stored content requested",
            metadata_json={"action": action, "from_status": doc.status},
        )

        # Run action
        try:
            await self._run_retry_ingest(db, doc=doc, request_id=request_id)

            await emit_document_event(
                db, tenant_id=tenant_id, document_id=document_id,
                event_type=RETRY_INGEST_SUCCEEDED,
                from_status=doc.status, to_status=doc.status,
                actor_user_id=actor_user_id,
                request_id=request_id,
                message=f"Retry ingest succeeded → {doc.status}",
                metadata_json={"action": action, "final_status": doc.status},
            )

            return ActionResult(
                document_id=document_id, action=action,
                accepted=True, status=doc.status,
                message="Retry ingest completed successfully",
                request_id=request_id,
            )

        except Exception as exc:
            logger.exception(
                "action.retry_ingest_failed doc_id=%s tenant_id=%s",
                document_id, tenant_id,
            )
            await emit_document_event(
                db, tenant_id=tenant_id, document_id=document_id,
                event_type=RETRY_INGEST_FAILED,
                actor_user_id=actor_user_id,
                request_id=request_id,
                message=f"Retry ingest failed: {str(exc)[:200]}",
                metadata_json={"action": action, "error": str(exc)[:200]},
            )

            return ActionResult(
                document_id=document_id, action=action,
                accepted=True, status=doc.status,
                message=f"Retry ingest failed: {str(exc)[:200]}",
                request_id=request_id,
            )

    async def _run_retry_ingest(
        self,
        db: AsyncSession,
        *,
        doc: Document,
        request_id: str,
    ) -> None:
        """Internal: re-run full pipeline from stored content_raw."""
        from app.services.document_service import DocumentService

        svc = DocumentService()

        # Re-process from raw content
        cleaned, chunks = svc._process_content(
            doc.content_raw,
            tenant_id=doc.tenant_id,
            document_id=doc.id,
            version_id=doc.checksum,
        )
        doc.content_text = cleaned

        # Transition to chunked
        old_status = doc.status
        if doc.status != CHUNKED:
            validate_transition(doc.status, CHUNKED)
            doc.status = CHUNKED
        await db.flush()

        await emit_document_event(
            db, tenant_id=doc.tenant_id, document_id=doc.id,
            event_type=RETRY_INGEST_STARTED,
            from_status=old_status, to_status=CHUNKED,
            request_id=request_id,
            message="Retry ingest: content re-processed, starting indexing",
        )

        # Re-embed and index
        await svc._embed_and_index(
            db, doc=doc, chunks=chunks, old_version=doc.checksum,
        )

    # ── Reindex ───────────────────────────────────────────────────────

    async def reindex(
        self,
        db: AsyncSession,
        *,
        document_id: int,
        tenant_id: str,
        actor_user_id: int | None = None,
    ) -> ActionResult:
        """
        Re-chunk + re-embed + re-index from stored content_text.

        Preconditions:
          - Document exists in tenant
          - Status in REINDEX_ALLOWED
          - content_text is not empty
        """
        request_id = str(uuid.uuid4())[:12]
        action = "reindex"

        doc = await self._repo.get_document_by_id(
            db, document_id=document_id, tenant_id=tenant_id,
        )
        if not doc:
            return ActionResult(
                document_id=document_id, action=action,
                accepted=False, status="not_found",
                message="Document not found", request_id=request_id,
            )

        if doc.status not in REINDEX_ALLOWED:
            return ActionResult(
                document_id=document_id, action=action,
                accepted=False, status=doc.status,
                message=f"Reindex not allowed in status '{doc.status}'. "
                        f"Allowed: {sorted(REINDEX_ALLOWED)}",
                request_id=request_id,
            )

        content_text = doc.content_text or doc.content_raw
        if not content_text or not content_text.strip():
            return ActionResult(
                document_id=document_id, action=action,
                accepted=False, status=doc.status,
                message="No content available for reindexing",
                request_id=request_id,
            )

        await emit_document_event(
            db, tenant_id=tenant_id, document_id=document_id,
            event_type=REINDEX_REQUESTED,
            actor_user_id=actor_user_id,
            request_id=request_id,
            message="Reindex document vectors requested",
            metadata_json={"action": action, "from_status": doc.status},
        )

        try:
            await self._run_reindex(db, doc=doc, content=content_text, request_id=request_id)

            await emit_document_event(
                db, tenant_id=tenant_id, document_id=document_id,
                event_type=REINDEX_SUCCEEDED,
                to_status=doc.status,
                actor_user_id=actor_user_id,
                request_id=request_id,
                message=f"Reindex succeeded → {doc.status}",
                metadata_json={"action": action, "final_status": doc.status},
            )

            return ActionResult(
                document_id=document_id, action=action,
                accepted=True, status=doc.status,
                message="Reindex completed successfully",
                request_id=request_id,
            )

        except Exception as exc:
            logger.exception(
                "action.reindex_failed doc_id=%s tenant_id=%s",
                document_id, tenant_id,
            )
            await emit_document_event(
                db, tenant_id=tenant_id, document_id=document_id,
                event_type=REINDEX_FAILED,
                actor_user_id=actor_user_id,
                request_id=request_id,
                message=f"Reindex failed: {str(exc)[:200]}",
                metadata_json={"action": action, "error": str(exc)[:200]},
            )

            return ActionResult(
                document_id=document_id, action=action,
                accepted=True, status=doc.status,
                message=f"Reindex failed: {str(exc)[:200]}",
                request_id=request_id,
            )

    async def _run_reindex(
        self,
        db: AsyncSession,
        *,
        doc: Document,
        content: str,
        request_id: str,
    ) -> None:
        """Internal: re-chunk, re-embed, re-index."""
        from app.services.document_service import DocumentService

        svc = DocumentService()

        # Re-chunk from cleaned content
        from app.nlp import get_chunker
        chunker = get_chunker()
        chunks = chunker.chunk(
            content,
            tenant_id=doc.tenant_id,
            document_id=doc.id,
            version_id=doc.checksum,
        )

        old_status = doc.status

        await emit_document_event(
            db, tenant_id=doc.tenant_id, document_id=doc.id,
            event_type=REINDEX_STARTED,
            from_status=old_status,
            request_id=request_id,
            message="Reindex: re-chunking and re-indexing vectors",
        )

        # Transition to chunked for re-indexing
        if doc.status == READY:
            validate_transition(doc.status, CHUNKED)
            doc.status = CHUNKED
            await db.flush()

        # Re-embed and index (replaces old vectors via old_version)
        await svc._embed_and_index(
            db, doc=doc, chunks=chunks, old_version=doc.checksum,
        )

    # ── Resynthesize ──────────────────────────────────────────────────

    async def resynthesize(
        self,
        db: AsyncSession,
        *,
        document_id: int,
        tenant_id: str,
        actor_user_id: int | None = None,
    ) -> ActionResult:
        """
        Regenerate synthesized child from original document.

        Preconditions:
          - Document exists in tenant
          - representation_type = 'original'
          - Has content_text or content_raw
        """
        request_id = str(uuid.uuid4())[:12]
        action = "resynthesize"

        doc = await self._repo.get_document_by_id(
            db, document_id=document_id, tenant_id=tenant_id,
        )
        if not doc:
            return ActionResult(
                document_id=document_id, action=action,
                accepted=False, status="not_found",
                message="Document not found", request_id=request_id,
            )

        if doc.representation_type not in RESYNTHESIZE_ALLOWED_TYPES:
            return ActionResult(
                document_id=document_id, action=action,
                accepted=False, status=doc.status,
                message=f"Resynthesize not allowed for '{doc.representation_type}' documents. "
                        f"Only original documents can be resynthesized.",
                request_id=request_id,
            )

        original_text = doc.content_text or doc.content_raw
        if not original_text or not original_text.strip():
            return ActionResult(
                document_id=document_id, action=action,
                accepted=False, status=doc.status,
                message="No content available for resynthesis",
                request_id=request_id,
            )

        await emit_document_event(
            db, tenant_id=tenant_id, document_id=document_id,
            event_type=RESYNTHESIZE_REQUESTED,
            actor_user_id=actor_user_id,
            request_id=request_id,
            message="Regenerate synthesized representation requested",
            metadata_json={"action": action},
        )

        try:
            child = await self._run_resynthesize(
                db, doc=doc, text=original_text, request_id=request_id,
            )

            child_id = child.id if child else None
            msg = (
                f"Resynthesize succeeded (child_id={child_id})"
                if child
                else "Resynthesize skipped (synthesis disabled or no output)"
            )

            await emit_document_event(
                db, tenant_id=tenant_id, document_id=document_id,
                event_type=RESYNTHESIZE_SUCCEEDED,
                actor_user_id=actor_user_id,
                request_id=request_id,
                message=msg,
                metadata_json={
                    "action": action,
                    "child_document_id": child_id,
                },
            )

            return ActionResult(
                document_id=document_id, action=action,
                accepted=True, status=doc.status,
                message=msg, request_id=request_id,
            )

        except Exception as exc:
            logger.exception(
                "action.resynthesize_failed doc_id=%s tenant_id=%s",
                document_id, tenant_id,
            )
            await emit_document_event(
                db, tenant_id=tenant_id, document_id=document_id,
                event_type=RESYNTHESIZE_FAILED,
                actor_user_id=actor_user_id,
                request_id=request_id,
                message=f"Resynthesize failed: {str(exc)[:200]}",
                metadata_json={"action": action, "error": str(exc)[:200]},
            )

            return ActionResult(
                document_id=document_id, action=action,
                accepted=True, status=doc.status,
                message=f"Resynthesize failed: {str(exc)[:200]}",
                request_id=request_id,
            )

    async def _run_resynthesize(
        self,
        db: AsyncSession,
        *,
        doc: Document,
        text: str,
        request_id: str,
    ) -> Document | None:
        """Internal: call synthesis orchestrator."""
        from app.services.synthesis_orchestrator import maybe_synthesize_child

        await emit_document_event(
            db, tenant_id=doc.tenant_id, document_id=doc.id,
            event_type=RESYNTHESIZE_STARTED,
            request_id=request_id,
            message="Resynthesize: calling synthesis engine",
        )

        child = await maybe_synthesize_child(
            db,
            original_doc=doc,
            original_text=text,
            tenant_id=doc.tenant_id,
        )
        return child
