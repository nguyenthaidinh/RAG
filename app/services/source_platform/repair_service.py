"""Source platform link repair service (Phase 2C).

Link-centric repair/reprocess for operator admin workflows.
Accepts a ``link_id``, reads the link to obtain its ``external_id``,
re-fetches the item from the source, maps it to canonical form, runs
the delta / upsert pipeline, and updates exactly that one link.

Does NOT:
  - Loop over multiple links
  - Call mark_missing / mark_all_missing
  - Create full sync runs
  - Touch the scheduler or coordinator
  - Affect any item other than the link identified by link_id

Caller owns the DB session and commit.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.onboarded_source import OnboardedSource
from app.repos.source_document_link_repo import SourceDocumentLinkRepo
from app.repos.source_platform_repo import SourcePlatformRepo
from app.schemas.source_platform import SourceRepairResponse
from app.services.source_platform.runtime_mapping import (
    row_to_runtime_config,
)
from app.services.source_platform.canonical_item import CanonicalKnowledgeItem
from app.services.source_platform.source_delta_resolver import (
    DeltaAction,
    SourceDeltaResolver,
)

logger = logging.getLogger(__name__)

# Map delta action → operator-friendly action label.
_ACTION_LABEL: dict[DeltaAction, str] = {
    DeltaAction.REACTIVATE: "reactivated",
    DeltaAction.UPDATE: "updated",
    DeltaAction.CREATE: "updated",  # shouldn't occur in repair, safe fallback
    DeltaAction.SKIP: "unchanged",  # only reached when force_reprocess=True
}


class SourcePlatformRepairService:
    """Link-centric repair/reprocess service for admin source platform.

    Receives a ``link_id`` (not an operator-typed ``external_id``),
    reads the link to get its ``external_id``, and runs the full
    fetch → map → delta → upsert → link-update cycle scoped to
    exactly that one link.
    """

    def __init__(
        self,
        source_repo: SourcePlatformRepo | None = None,
        link_repo: SourceDocumentLinkRepo | None = None,
        delta_resolver: SourceDeltaResolver | None = None,
    ) -> None:
        self._source_repo = source_repo or SourcePlatformRepo()
        self._link_repo = link_repo or SourceDocumentLinkRepo()
        self._delta_resolver = delta_resolver or SourceDeltaResolver()

    # ── Public API ───────────────────────────────────────────────

    async def repair_link(
        self,
        db: AsyncSession,
        *,
        tenant_id: str,
        source_id: int,
        link_id: int,
        force_reprocess: bool = False,
    ) -> SourceRepairResponse | None:
        """Repair / reprocess exactly one source link by link_id.

        Returns ``None`` only if the source itself is not found (404).
        Link-not-found and all other soft failures return a response
        with ``ok=False`` and a descriptive message — no mutations.
        """
        now = datetime.now(timezone.utc)
        warnings: list[str] = []

        # ── A. Lookup source (tenant-scoped) ─────────────────────
        source = await self._source_repo.get_source(
            db, tenant_id=tenant_id, source_id=source_id
        )
        if not source:
            return None

        config = self._row_to_config(source)

        # ── B. Lookup link by link_id (tenant + source scoped) ───
        link = await self._link_repo.get_link_by_id(
            db,
            tenant_id=tenant_id,
            onboarded_source_id=source.id,
            link_id=link_id,
        )
        if not link:
            return SourceRepairResponse(
                source_id=source.id,
                source_key=source.source_key,
                link_id=link_id,
                external_id="",
                ok=False,
                action_taken="failed",
                previous_link_status=None,
                new_link_status=None,
                had_document_before=False,
                raw_item_found=False,
                canonical_mapped=False,
                message=(
                    f"Link not found: link_id={link_id} "
                    f"source_id={source_id}"
                ),
                repaired_at=now,
            )

        external_id = link.external_id
        previous_link_status = link.status
        had_document_before = link.document_id is not None

        # ── C. Build connector ───────────────────────────────────
        try:
            connector = config.build_connector()
        except ValueError as exc:
            return SourceRepairResponse(
                source_id=source.id,
                source_key=source.source_key,
                link_id=link_id,
                external_id=external_id,
                ok=False,
                action_taken="failed",
                previous_link_status=previous_link_status,
                new_link_status=previous_link_status,
                had_document_before=had_document_before,
                raw_item_found=False,
                canonical_mapped=False,
                message=f"Cannot build connector: {exc}",
                repaired_at=now,
            )

        # ── D. Fetch raw detail from source ──────────────────────
        try:
            raw_detail = await connector.fetch_item_detail(
                tenant_id=tenant_id,
                source_key=source.source_key,
                external_id=external_id,
            )
        except Exception as exc:
            logger.warning(
                "source_platform.repair.fetch_error "
                "source_id=%d link_id=%d external_id=%s: %s",
                source.id,
                link_id,
                external_id,
                str(exc),
                exc_info=True,
            )
            return SourceRepairResponse(
                source_id=source.id,
                source_key=source.source_key,
                link_id=link_id,
                external_id=external_id,
                ok=False,
                action_taken="failed",
                previous_link_status=previous_link_status,
                new_link_status=previous_link_status,
                had_document_before=had_document_before,
                raw_item_found=False,
                canonical_mapped=False,
                message=f"Fetch failed: {str(exc)[:500]}",
                repaired_at=now,
            )

        # ── E. Item not found at source ──────────────────────────
        # Do NOT call mark_missing, do NOT sweep, do NOT mutate the link.
        if raw_detail is None:
            return SourceRepairResponse(
                source_id=source.id,
                source_key=source.source_key,
                link_id=link_id,
                external_id=external_id,
                ok=False,
                action_taken="failed",
                previous_link_status=previous_link_status,
                new_link_status=previous_link_status,
                had_document_before=had_document_before,
                raw_item_found=False,
                canonical_mapped=False,
                message=(
                    f"Item not found at source for "
                    f"external_id='{external_id}' — no changes made"
                ),
                repaired_at=now,
            )

        # ── F. Map to canonical ──────────────────────────────────
        try:
            canonical = connector.map_to_canonical_item(
                source_key=source.source_key,
                raw_detail=raw_detail,
            )
        except Exception as exc:
            logger.warning(
                "source_platform.repair.map_error "
                "source_id=%d link_id=%d external_id=%s: %s",
                source.id,
                link_id,
                external_id,
                str(exc),
                exc_info=True,
            )
            return SourceRepairResponse(
                source_id=source.id,
                source_key=source.source_key,
                link_id=link_id,
                external_id=external_id,
                ok=False,
                action_taken="failed",
                previous_link_status=previous_link_status,
                new_link_status=previous_link_status,
                had_document_before=had_document_before,
                raw_item_found=True,
                canonical_mapped=False,
                message=f"Canonical mapping error: {str(exc)[:500]}",
                repaired_at=now,
            )

        if canonical is None:
            return SourceRepairResponse(
                source_id=source.id,
                source_key=source.source_key,
                link_id=link_id,
                external_id=external_id,
                ok=False,
                action_taken="failed",
                previous_link_status=previous_link_status,
                new_link_status=previous_link_status,
                had_document_before=had_document_before,
                raw_item_found=True,
                canonical_mapped=False,
                message=(
                    "Canonical mapping returned None — item lacks "
                    "meaningful textual content or external_id"
                ),
                repaired_at=now,
            )

        # ── G. Delta resolve against the existing link ───────────
        # We already have the link from step B — pass it directly.
        delta = self._delta_resolver.resolve(
            existing_link=link,
            canonical=canonical,
        )

        # ── H. Handle SKIP / force_reprocess ─────────────────────
        if force_reprocess and delta.action == DeltaAction.SKIP:
            action_label = "force_reprocessed"
            needs_upsert = True
            warnings.append(
                "force_reprocess=true: overriding SKIP "
                f"(reason={delta.reason}) → will re-upsert document"
            )
        elif delta.action == DeltaAction.SKIP:
            # Content unchanged and not forced — touch seen only, no upsert.
            await self._link_repo.touch_seen(db, link, now=now)
            return SourceRepairResponse(
                source_id=source.id,
                source_key=source.source_key,
                link_id=link_id,
                external_id=external_id,
                ok=True,
                action_taken="unchanged",
                previous_link_status=previous_link_status,
                new_link_status=link.status,
                document_id=link.document_id,
                # Transitional fallback: prefer version_id, fall back to checksum
                document_version_id=link.document_version_id or link.content_checksum,
                content_checksum=canonical.effective_checksum(),
                had_document_before=had_document_before,
                raw_item_found=True,
                canonical_mapped=True,
                message="Content unchanged — no upsert needed",
                repaired_at=now,
            )
        else:
            action_label = _ACTION_LABEL[delta.action]
            needs_upsert = True

        # ── I. Upsert document ───────────────────────────────────
        if needs_upsert:
            try:
                doc, doc_action, _processed = await self._upsert_document(
                    db,
                    tenant_id=tenant_id,
                    canonical=canonical,
                )
            except Exception as exc:
                logger.warning(
                    "source_platform.repair.upsert_error "
                    "source_id=%d link_id=%d external_id=%s: %s",
                    source.id,
                    link_id,
                    external_id,
                    str(exc),
                    exc_info=True,
                )
                return SourceRepairResponse(
                    source_id=source.id,
                    source_key=source.source_key,
                    link_id=link_id,
                    external_id=external_id,
                    ok=False,
                    action_taken="failed",
                    previous_link_status=previous_link_status,
                    new_link_status=previous_link_status,
                    had_document_before=had_document_before,
                    raw_item_found=True,
                    canonical_mapped=True,
                    message=f"Document upsert failed: {str(exc)[:500]}",
                    repaired_at=now,
                )

            if doc_action == "noop":
                warnings.append(
                    "DocumentService.upsert returned 'noop' — "
                    "document content already identical"
                )

        # ── J. Update the link ───────────────────────────────────
        checksum = canonical.effective_checksum()

        if delta.action == DeltaAction.REACTIVATE:
            await self._link_repo.reactivate(db, link, now=now)
            await self._link_repo.touch_synced(
                db,
                link,
                now=now,
                content_checksum=checksum,
                document_id=doc.id,
                document_version_id=getattr(doc, "version_id", None),
                external_uri=canonical.source_uri,
                remote_updated_at=canonical.updated_at,
            )
        else:
            # UPDATE or force_reprocessed (SKIP overridden)
            await self._link_repo.touch_synced(
                db,
                link,
                now=now,
                content_checksum=checksum,
                document_id=doc.id,
                document_version_id=getattr(doc, "version_id", None),
                external_uri=canonical.source_uri,
                remote_updated_at=canonical.updated_at,
            )

        logger.info(
            "source_platform.repair.ok "
            "source_id=%d link_id=%d external_id=%s action=%s doc_id=%s",
            source.id,
            link_id,
            external_id,
            action_label,
            doc.id,
        )

        return SourceRepairResponse(
            source_id=source.id,
            source_key=source.source_key,
            link_id=link_id,
            external_id=external_id,
            ok=True,
            action_taken=action_label,
            previous_link_status=previous_link_status,
            new_link_status=link.status,
            document_id=doc.id,
            document_version_id=getattr(doc, "version_id", None),
            content_checksum=checksum,
            had_document_before=had_document_before,
            raw_item_found=True,
            canonical_mapped=True,
            message=f"Link repair completed: {action_label}",
            warnings=warnings,
            repaired_at=now,
        )

    # ── Helpers ──────────────────────────────────────────────────

    async def _upsert_document(
        self,
        db: AsyncSession,
        *,
        tenant_id: str,
        canonical: CanonicalKnowledgeItem,
    ) -> tuple[Any, str, bool]:
        """Upsert document via DocumentService.

        Uses the shared ``build_upsert_payload()`` helper to ensure
        identical ingest payloads across all source platform flows.
        """
        from app.services.document_service import DocumentService
        from app.services.source_platform.runtime_mapping import (
            build_upsert_payload,
        )

        doc_svc = DocumentService()
        payload = build_upsert_payload(canonical, tenant_id=tenant_id)
        return await doc_svc.upsert(db, **payload)

    @staticmethod
    def _row_to_config(row: OnboardedSource) -> OnboardedSourceConfig:
        """Map a DB row to runtime config.

        Delegates to the shared ``row_to_runtime_config()`` helper
        to ensure all source platform services use identical mapping.
        """
        return row_to_runtime_config(row)
