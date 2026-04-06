"""Source platform single-item action service (Phase 2B).

Safe, scoped, single-item sync for operator admin workflows.
Fetches exactly one item by ``external_id``, resolves delta,
upserts the document, and updates/creates the source link.

Does NOT:
  - Loop over multiple items
  - Call mark_missing / mark_all_missing
  - Create full sync runs
  - Touch the scheduler or coordinator
  - Affect any item other than the one specified

Caller owns the DB session and commit.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.onboarded_source import OnboardedSource
from app.db.models.source_document_link import SourceDocumentLink
from app.repos.source_document_link_repo import SourceDocumentLinkRepo
from app.repos.source_platform_repo import SourcePlatformRepo
from app.schemas.source_platform import SyncOneResponse
from app.services.source_platform.runtime_mapping import (
    row_to_runtime_config,
)
from app.services.source_platform.canonical_item import CanonicalKnowledgeItem
from app.services.source_platform.source_delta_resolver import (
    DeltaAction,
    SourceDeltaResolver,
)

logger = logging.getLogger(__name__)


class SourcePlatformActionService:
    """Single-item sync action service for admin source platform.

    Follows the same fetch → map → delta → upsert → link pattern
    as ``SourceSyncService._sync_single_item_delta``, but scoped
    to exactly one item and invoked on-demand by the operator.
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

    async def sync_one(
        self,
        db: AsyncSession,
        *,
        tenant_id: str,
        source_id: int,
        external_id: str,
        force_reprocess: bool = False,
    ) -> SyncOneResponse | None:
        """Sync exactly one item by external_id.

        Returns ``None`` only if the source itself is not found (404).
        All other outcomes (item not found at source, mapping failure,
        upsert error) are returned as a response with ``ok=False``.
        """
        now = datetime.now(timezone.utc)
        warnings: list[str] = []

        # ── A. Lookup source ─────────────────────────────────────
        source = await self._source_repo.get_source(
            db, tenant_id=tenant_id, source_id=source_id
        )
        if not source:
            return None

        config = self._row_to_config(source)

        # ── B. Build connector ───────────────────────────────────
        try:
            connector = config.build_connector()
        except ValueError as exc:
            return SyncOneResponse(
                source_id=source.id,
                source_key=source.source_key,
                external_id=external_id,
                ok=False,
                action_taken="failed",
                message=f"Cannot build connector: {exc}",
                synced_at=now,
            )

        # ── C. Fetch raw detail ──────────────────────────────────
        try:
            raw_detail = await connector.fetch_item_detail(
                tenant_id=tenant_id,
                source_key=source.source_key,
                external_id=external_id,
            )
        except Exception as exc:
            logger.warning(
                "source_platform.action.sync_one fetch_error "
                "source_id=%d external_id=%s: %s",
                source.id,
                external_id,
                str(exc),
                exc_info=True,
            )
            return SyncOneResponse(
                source_id=source.id,
                source_key=source.source_key,
                external_id=external_id,
                ok=False,
                action_taken="failed",
                message=f"Fetch failed: {str(exc)[:500]}",
                synced_at=now,
            )

        # ── D. Item not found at source ──────────────────────────
        if raw_detail is None:
            return SyncOneResponse(
                source_id=source.id,
                source_key=source.source_key,
                external_id=external_id,
                ok=False,
                action_taken="failed",
                message=(
                    f"Item not found at source for "
                    f"external_id='{external_id}'"
                ),
                synced_at=now,
            )

        # ── E. Map to canonical ──────────────────────────────────
        try:
            canonical = connector.map_to_canonical_item(
                source_key=source.source_key,
                raw_detail=raw_detail,
            )
        except Exception as exc:
            logger.warning(
                "source_platform.action.sync_one map_error "
                "source_id=%d external_id=%s: %s",
                source.id,
                external_id,
                str(exc),
                exc_info=True,
            )
            return SyncOneResponse(
                source_id=source.id,
                source_key=source.source_key,
                external_id=external_id,
                ok=False,
                action_taken="failed",
                message=f"Canonical mapping error: {str(exc)[:500]}",
                synced_at=now,
            )

        if canonical is None:
            return SyncOneResponse(
                source_id=source.id,
                source_key=source.source_key,
                external_id=external_id,
                ok=False,
                action_taken="failed",
                message=(
                    "Canonical mapping returned None — item lacks "
                    "meaningful textual content or external_id"
                ),
                synced_at=now,
            )

        # ── F. Lookup existing link ──────────────────────────────
        existing_link = await self._link_repo.get_by_external_id(
            db,
            tenant_id=tenant_id,
            onboarded_source_id=source.id,
            external_id=external_id,
        )
        had_existing_link = existing_link is not None
        previous_link_status = (
            existing_link.status if existing_link else None
        )

        # ── G. Delta resolve ─────────────────────────────────────
        delta = self._delta_resolver.resolve(
            existing_link=existing_link,
            canonical=canonical,
        )

        # ── H. Handle force_reprocess override ───────────────────
        if force_reprocess and delta.action == DeltaAction.SKIP:
            # Override SKIP → force upsert
            action_label = "force_reprocessed"
            needs_upsert = True
            warnings.append(
                "force_reprocess=true: overriding SKIP "
                f"(reason={delta.reason}) → will re-upsert document"
            )
        elif delta.action == DeltaAction.SKIP:
            # Unchanged, just touch last_seen_at
            await self._link_repo.touch_seen(
                db, existing_link, now=now
            )
            return SyncOneResponse(
                source_id=source.id,
                source_key=source.source_key,
                external_id=external_id,
                ok=True,
                action_taken="unchanged",
                content_checksum=canonical.effective_checksum(),
                had_existing_link=had_existing_link,
                previous_link_status=previous_link_status,
                canonical_mapped=True,
                document_id=(
                    existing_link.document_id if existing_link else None
                ),
                message="Content unchanged — no upsert needed",
                synced_at=now,
            )
        else:
            action_label = delta.action.value  # create / update / reactivate
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
                    "source_platform.action.sync_one upsert_error "
                    "source_id=%d external_id=%s: %s",
                    source.id,
                    external_id,
                    str(exc),
                    exc_info=True,
                )
                return SyncOneResponse(
                    source_id=source.id,
                    source_key=source.source_key,
                    external_id=external_id,
                    ok=False,
                    action_taken="failed",
                    canonical_mapped=True,
                    had_existing_link=had_existing_link,
                    previous_link_status=previous_link_status,
                    message=f"Document upsert failed: {str(exc)[:500]}",
                    synced_at=now,
                )

            if doc_action == "noop":
                warnings.append(
                    "DocumentService.upsert returned 'noop' — "
                    "document content already identical"
                )

        # ── J. Update / create link ──────────────────────────────
        checksum = canonical.effective_checksum()

        if delta.action == DeltaAction.CREATE:
            new_link = SourceDocumentLink(
                tenant_id=tenant_id,
                onboarded_source_id=source.id,
                source_key=source.source_key,
                external_id=external_id,
                external_uri=canonical.source_uri,
                document_id=doc.id,
                document_version_id=getattr(doc, "version_id", None),
                content_checksum=checksum,
                remote_updated_at=canonical.updated_at,
                last_seen_at=now,
                last_synced_at=now,
                status="active",
            )
            self._link_repo.add(db, new_link)
            await db.flush()

        elif delta.action == DeltaAction.REACTIVATE:
            await self._link_repo.reactivate(
                db, existing_link, now=now
            )
            await self._link_repo.touch_synced(
                db,
                existing_link,
                now=now,
                content_checksum=checksum,
                document_id=doc.id,
                document_version_id=getattr(doc, "version_id", None),
                external_uri=canonical.source_uri,
                remote_updated_at=canonical.updated_at,
            )

        elif delta.action in (DeltaAction.UPDATE, DeltaAction.SKIP):
            # UPDATE or force_reprocessed (SKIP overridden)
            await self._link_repo.touch_synced(
                db,
                existing_link,
                now=now,
                content_checksum=checksum,
                document_id=doc.id,
                document_version_id=getattr(doc, "version_id", None),
                external_uri=canonical.source_uri,
                remote_updated_at=canonical.updated_at,
            )

        logger.info(
            "source_platform.action.sync_one ok "
            "source_id=%d external_id=%s action=%s doc_id=%s",
            source.id,
            external_id,
            action_label,
            doc.id,
        )

        return SyncOneResponse(
            source_id=source.id,
            source_key=source.source_key,
            external_id=external_id,
            ok=True,
            action_taken=action_label,
            document_id=doc.id,
            document_version_id=getattr(doc, "version_id", None),
            content_checksum=checksum,
            had_existing_link=had_existing_link,
            previous_link_status=previous_link_status,
            canonical_mapped=True,
            message=f"Single-item sync completed: {action_label}",
            warnings=warnings,
            synced_at=now,
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
