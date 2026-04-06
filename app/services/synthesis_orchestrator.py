"""
Synthesis orchestrator (Phase 9.0 — Step 3).

Thin orchestration layer that wires the standalone DocumentSynthesisService
into the ingest flow.  After an original document is successfully ingested,
this module:

1. Calls DocumentSynthesisService to produce a structured markdown version.
2. Upserts a child document with representation_type='synthesized' and
   parent_document_id pointing to the original.

Design principles:
- **Fail-open**: synthesis failure never breaks original ingest.
- **Idempotent**: uses deterministic (source, external_id) derived from
  the original to avoid duplicates on re-ingest.
- **Minimal**: delegates to existing DocumentService.upsert for the child,
  reusing the full chunk → embed → index pipeline.
- **No side-effects on disabled**: returns immediately when SYNTHESIS_ENABLED
  is False.
"""
from __future__ import annotations

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.models.document import Document
from app.services.document_event_emitter import (
    emit_document_event,
    DOCUMENT_SYNTHESIZED_CHILD_CREATED,
)

logger = logging.getLogger(__name__)

# Suffix appended to original external_id to form synthesized child identity.
# Using external_id suffix (not source suffix) because:
# - keeps same source grouping for admin queries
# - clearly marks the external_id as derived
# - stable across re-ingests (deterministic)
_SYNTH_EXTERNAL_ID_SUFFIX = "::synthesized:v1"


async def maybe_synthesize_child(
    db: AsyncSession,
    *,
    original_doc: Document,
    original_text: str,
    tenant_id: str,
) -> Document | None:
    """
    Best-effort synthesis of a child document from the original.

    Returns the synthesized Document on success, None if skipped or failed.
    Never raises — all errors are caught and logged.

    Args:
        db: Active async session (caller owns transaction).
        original_doc: The already-persisted original Document.
        original_text: Extracted text of the original (used as synthesis input).
        tenant_id: Tenant scope.
    """
    # ── Gate: skip if disabled ───────────────────────────────────────
    if not settings.SYNTHESIS_ENABLED:
        return None

    # ── Gate: only synthesize originals ──────────────────────────────
    if getattr(original_doc, "representation_type", "original") != "original":
        return None

    # ── Gate: skip if no meaningful text ─────────────────────────────
    if not original_text or not original_text.strip():
        return None

    try:
        return await _do_synthesize(
            db,
            original_doc=original_doc,
            original_text=original_text,
            tenant_id=tenant_id,
        )
    except Exception:
        # Fail-open: never break original ingest
        logger.warning(
            "synthesis.orchestrator_failed tenant_id=%s doc_id=%s",
            tenant_id, original_doc.id,
            exc_info=True,
        )
        return None


async def _do_synthesize(
    db: AsyncSession,
    *,
    original_doc: Document,
    original_text: str,
    tenant_id: str,
) -> Document | None:
    """
    Internal: run synthesis + upsert child. May raise on unexpected errors.
    """
    from app.services.document_synthesis_service import (
        DocumentSynthesisError,
        DocumentSynthesisService,
    )
    from app.services.document_service import DocumentService

    # ── 1. Call synthesis engine ──────────────────────────────────────
    svc = DocumentSynthesisService()

    try:
        result = await svc.synthesize_document(
            title=original_doc.title,
            content_text=original_text,
            document_id=original_doc.id,
            tenant_id=tenant_id,
        )
    except DocumentSynthesisError as exc:
        logger.warning(
            "synthesis.engine_failed tenant_id=%s doc_id=%s reason=%s",
            tenant_id, original_doc.id, str(exc)[:200],
        )
        return None

    # ── 2. Build synthesized child identity ──────────────────────────
    synth_source = original_doc.source
    synth_external_id = original_doc.external_id + _SYNTH_EXTERNAL_ID_SUFFIX

    synth_metadata = {
        "synthesis": {
            "provider": result.provider,
            "model": result.model,
            "prompt_version": result.prompt_version,
            "parent_document_id": original_doc.id,
            "input_chars": result.input_chars,
            "output_chars": result.output_chars,
        },
        "ingest_via": "synthesis",
    }

    # ── 3. Upsert synthesized child through existing pipeline ────────
    doc_svc = DocumentService()

    child_doc, action, changed = await doc_svc.upsert(
        db=db,
        tenant_id=tenant_id,
        source=synth_source,
        external_id=synth_external_id,
        content=result.content_markdown,
        title=f"{original_doc.title or 'Document'} (Synthesized)" if original_doc.title else "Synthesized Document",
        metadata=synth_metadata,
        representation_type="synthesized",
        parent_document_id=original_doc.id,
    )

    logger.info(
        "synthesis.child_created tenant_id=%s parent_id=%s child_id=%s "
        "action=%s output_chars=%d",
        tenant_id, original_doc.id, child_doc.id,
        action, result.output_chars,
    )

    # Phase 2A: emit synthesized child event on the parent document
    await emit_document_event(
        db,
        tenant_id=tenant_id,
        document_id=original_doc.id,
        event_type=DOCUMENT_SYNTHESIZED_CHILD_CREATED,
        message=f"Synthesized child document created (child_id={child_doc.id})",
        metadata_json={
            "child_document_id": child_doc.id,
            "provider": result.provider,
            "model": result.model,
        },
    )

    return child_doc
