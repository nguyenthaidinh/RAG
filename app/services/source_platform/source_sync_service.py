"""
Source sync service (Phase 8 — Delta-Aware Sync).

Orchestrates the flow::

    connector.fetch_item_refs()
    → connector.fetch_item_detail()
    → connector.map_to_canonical_item()
    → delta resolve (skip / create / update / reactivate)
    → DocumentService.upsert()  (only when needed)
    → update link table
    → sweep missing items

``SourceSyncService`` resolves the connector, iterates items with
delta-awareness, and delegates persistence to the existing document
pipeline.  It does NOT own scheduling, job state, or health tracking.

Backward compatibility:
  When ``onboarded_source_id`` is None, the service falls back to the
  original non-delta behavior (always upsert, no link tracking).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.source_platform.canonical_item import CanonicalKnowledgeItem
from app.services.source_platform.source_registry import SourceRegistry

logger = logging.getLogger(__name__)


# ── Result model ─────────────────────────────────────────────────────


@dataclass
class SyncResult:
    """Summary counters for a single sync run.

    Phase 8 metrics are granular.  Backward-compatible properties
    (``fetched_count``, ``upserted_count``, etc.) are preserved so
    callers that use the old field names continue to work.
    """

    discovered_count: int = 0
    created_count: int = 0
    updated_count: int = 0
    unchanged_count: int = 0
    missing_count: int = 0
    error_count: int = 0
    reactivated_count: int = 0
    errors: list[dict[str, str]] = field(default_factory=list)

    # ── Backward-compatible aliases ──────────────────────────────

    @property
    def fetched_count(self) -> int:
        return self.discovered_count

    @property
    def mapped_count(self) -> int:
        return (
            self.created_count
            + self.updated_count
            + self.unchanged_count
            + self.reactivated_count
        )

    @property
    def upserted_count(self) -> int:
        return self.created_count + self.updated_count + self.reactivated_count

    @property
    def skipped_count(self) -> int:
        return self.unchanged_count

    @property
    def failed_count(self) -> int:
        return self.error_count


# ── Service ──────────────────────────────────────────────────────────

_MAX_ERROR_LOG_ITEMS = 50


class SourceSyncService:
    """
    Orchestrates syncing items from a source connector into the
    document pipeline via ``DocumentService.upsert()``, with
    optional delta-aware link tracking.

    Parameters
    ----------
    registry : SourceRegistry
        Registry from which connectors are resolved.
    document_service : DocumentService
        The production ``DocumentService`` instance used as the
        persistence sink.
    link_repo : SourceDocumentLinkRepo | None
        Repository for source ↔ document link tracking.
        When None, delta resolution is skipped (legacy behavior).
    delta_resolver : SourceDeltaResolver | None
        Resolver for determining sync action per item.
        When None, a default instance is created lazily.
    """

    def __init__(
        self,
        registry: SourceRegistry,
        document_service: Any,  # DocumentService — late-bound to avoid circular imports
        *,
        link_repo: Any | None = None,
        delta_resolver: Any | None = None,
    ) -> None:
        self._registry = registry
        self._doc_svc = document_service
        self._link_repo = link_repo
        self._delta_resolver = delta_resolver

    def _get_link_repo(self):
        if self._link_repo is None:
            from app.repos.source_document_link_repo import (
                SourceDocumentLinkRepo,
            )
            self._link_repo = SourceDocumentLinkRepo()
        return self._link_repo

    def _get_delta_resolver(self):
        if self._delta_resolver is None:
            from app.services.source_platform.source_delta_resolver import (
                SourceDeltaResolver,
            )
            self._delta_resolver = SourceDeltaResolver()
        return self._delta_resolver

    # ── Public API ───────────────────────────────────────────────

    async def sync_items(
        self,
        db: AsyncSession,
        *,
        tenant_id: str,
        provider_name: str,
        source_key: str,
        params: dict[str, Any] | None = None,
        item_refs: list[dict[str, Any]] | None = None,
        onboarded_source_id: int | None = None,
    ) -> SyncResult:
        """
        Sync items from a registered source connector.

        When ``onboarded_source_id`` is provided, delta-aware sync
        is used: items are resolved against the link table, and only
        changed/new items are upserted.  A missing sweep runs at the end.

        When ``onboarded_source_id`` is None, falls back to the legacy
        always-upsert behavior (no link tracking).

        Args:
            db:            Async DB session (caller owns the transaction).
            tenant_id:     Tenant scope.
            provider_name: Connector name as registered in the registry.
            source_key:    Logical source identifier (maps to
                           ``DocumentService.upsert(source=...)``).
            params:        Optional connector-specific query parameters
                           passed to ``fetch_item_refs()``.
            item_refs:     If provided, skip ``fetch_item_refs()`` and
                           iterate over this list directly.
            onboarded_source_id:
                           DB id of the OnboardedSource. When provided,
                           enables delta-aware sync with link tracking.

        Returns:
            A ``SyncResult`` with counters and error summaries.
        """
        result = SyncResult()

        connector = self._registry.get_required(provider_name)

        # ── 1. Resolve item refs ─────────────────────────────────
        if item_refs is None:
            try:
                item_refs = await connector.fetch_item_refs(
                    tenant_id=tenant_id,
                    source_key=source_key,
                    params=params,
                )
            except Exception:
                logger.exception(
                    "source_sync.fetch_refs_failed provider=%s source_key=%s tenant=%s",
                    provider_name,
                    source_key,
                    tenant_id,
                )
                result.error_count += 1
                result.errors.append(
                    {"phase": "fetch_refs", "error": "Failed to fetch item refs"}
                )
                return result

        result.discovered_count = len(item_refs or [])

        # Empty result from a successful fetch is a valid snapshot:
        # the source has zero items.  In delta mode we must still sweep
        # existing links to mark them as missing.  In legacy mode there
        # is nothing to do.
        if not item_refs:
            if onboarded_source_id is not None:
                now = datetime.now(timezone.utc)
                try:
                    link_repo = self._get_link_repo()
                    missing_count = await link_repo.mark_all_missing(
                        db,
                        tenant_id=tenant_id,
                        onboarded_source_id=onboarded_source_id,
                        now=now,
                    )
                    result.missing_count = missing_count
                except Exception:
                    logger.exception(
                        "source_sync.missing_sweep_failed provider=%s source_key=%s tenant=%s",
                        provider_name,
                        source_key,
                        tenant_id,
                    )
                    result.error_count += 1
                    self._record_error(
                        result,
                        "missing_sweep",
                        "",
                        "Failed to sweep missing items on empty fetch",
                    )
            return result

        logger.info(
            "source_sync.start provider=%s source_key=%s tenant=%s items=%d delta=%s",
            provider_name,
            source_key,
            tenant_id,
            result.discovered_count,
            onboarded_source_id is not None,
        )

        # ── 2. Determine sync mode ──────────────────────────────
        use_delta = onboarded_source_id is not None
        now = datetime.now(timezone.utc)
        seen_external_ids: set[str] = set()

        # ── 3. Process each item (fail-safe per item) ────────────
        for ref in item_refs:
            external_id = ref.get("external_id", "")
            if not external_id:
                result.error_count += 1
                self._record_error(
                    result, "missing_id", external_id, "ref has no external_id"
                )
                continue

            try:
                if use_delta:
                    # onboarded_source_id is guaranteed non-None here
                    assert onboarded_source_id is not None
                    await self._sync_single_item_delta(
                        db,
                        connector=connector,
                        tenant_id=tenant_id,
                        source_key=source_key,
                        onboarded_source_id=onboarded_source_id,
                        external_id=external_id,
                        ref=ref,
                        result=result,
                        now=now,
                        seen_external_ids=seen_external_ids,
                    )
                else:
                    await self._sync_single_item_legacy(
                        db,
                        connector=connector,
                        tenant_id=tenant_id,
                        source_key=source_key,
                        external_id=external_id,
                        ref=ref,
                        result=result,
                    )
            except Exception:
                result.error_count += 1
                logger.exception(
                    "source_sync.item_failed provider=%s source_key=%s external_id=%s",
                    provider_name,
                    source_key,
                    external_id,
                )
                self._record_error(
                    result,
                    "item_exception",
                    external_id,
                    "Unhandled exception during sync",
                )

        # ── 4. Missing sweep (delta mode only) ───────────────────
        if use_delta and seen_external_ids:
            try:
                link_repo = self._get_link_repo()
                missing_count = await link_repo.mark_missing(
                    db,
                    tenant_id=tenant_id,
                    onboarded_source_id=onboarded_source_id,
                    seen_external_ids=seen_external_ids,
                    now=now,
                )
                result.missing_count = missing_count
            except Exception:
                logger.exception(
                    "source_sync.missing_sweep_failed provider=%s source_key=%s tenant=%s",
                    provider_name,
                    source_key,
                    tenant_id,
                )
                result.error_count += 1
                self._record_error(
                    result,
                    "missing_sweep",
                    "",
                    "Failed to sweep missing items",
                )

        logger.info(
            "source_sync.done provider=%s source_key=%s tenant=%s "
            "discovered=%d created=%d updated=%d unchanged=%d "
            "reactivated=%d missing=%d errors=%d",
            provider_name,
            source_key,
            tenant_id,
            result.discovered_count,
            result.created_count,
            result.updated_count,
            result.unchanged_count,
            result.reactivated_count,
            result.missing_count,
            result.error_count,
        )

        return result

    # ── Delta-aware item sync ────────────────────────────────────

    async def _sync_single_item_delta(
        self,
        db: AsyncSession,
        *,
        connector: Any,
        tenant_id: str,
        source_key: str,
        onboarded_source_id: int,
        external_id: str,
        ref: dict[str, Any],
        result: SyncResult,
        now: datetime,
        seen_external_ids: set[str],
    ) -> None:
        """Fetch, map, delta-resolve, and conditionally upsert one item."""
        from app.db.models.source_document_link import SourceDocumentLink
        from app.services.source_platform.source_delta_resolver import (
            DeltaAction,
        )

        link_repo = self._get_link_repo()
        resolver = self._get_delta_resolver()

        seen_external_ids.add(external_id)

        # Fetch detail
        detail = await connector.fetch_item_detail(
            tenant_id=tenant_id,
            source_key=source_key,
            external_id=external_id,
            ref=ref,
        )
        if detail is None:
            result.unchanged_count += 1
            logger.debug(
                "source_sync.detail_none external_id=%s source_key=%s",
                external_id,
                source_key,
            )
            return

        # Map to canonical
        canonical = connector.map_to_canonical_item(
            source_key=source_key,
            raw_detail=detail,
        )
        if canonical is None:
            result.error_count += 1
            self._record_error(
                result, "map_failed", external_id,
                "map_to_canonical_item returned None",
            )
            logger.debug(
                "source_sync.map_none external_id=%s source_key=%s",
                external_id,
                source_key,
            )
            return

        # Lookup existing link
        existing_link = await link_repo.get_by_external_id(
            db,
            tenant_id=tenant_id,
            onboarded_source_id=onboarded_source_id,
            external_id=external_id,
        )

        # Delta resolve
        delta = resolver.resolve(
            existing_link=existing_link,
            canonical=canonical,
        )

        # ── SKIP: just touch last_seen_at ────────────────────────
        if delta.action == DeltaAction.SKIP:
            result.unchanged_count += 1
            await link_repo.touch_seen(db, delta.existing_link, now=now)
            logger.debug(
                "source_sync.skip external_id=%s reason=%s",
                external_id,
                delta.reason,
            )
            return

        # ── CREATE / UPDATE / REACTIVATE: upsert document ────────
        payload = self._build_upsert_payload(canonical, tenant_id=tenant_id)
        doc, action, _processed = await self._doc_svc.upsert(db, **payload)

        if delta.action == DeltaAction.CREATE:
            # Create new link
            new_link = SourceDocumentLink(
                tenant_id=tenant_id,
                onboarded_source_id=onboarded_source_id,
                source_key=source_key,
                external_id=external_id,
                external_uri=canonical.source_uri,
                document_id=doc.id,
                document_version_id=getattr(doc, "version_id", None),
                content_checksum=canonical.effective_checksum(),
                remote_updated_at=canonical.updated_at,
                last_seen_at=now,
                last_synced_at=now,
                status="active",
            )
            link_repo.add(db, new_link)
            await db.flush()
            result.created_count += 1

        elif delta.action == DeltaAction.REACTIVATE:
            # Reactivate missing link
            await link_repo.reactivate(db, delta.existing_link, now=now)
            await link_repo.touch_synced(
                db,
                delta.existing_link,
                now=now,
                content_checksum=canonical.effective_checksum(),
                document_id=doc.id,
                document_version_id=getattr(doc, "version_id", None),
                external_uri=canonical.source_uri,
                remote_updated_at=canonical.updated_at,
            )
            result.reactivated_count += 1

        else:
            # UPDATE
            await link_repo.touch_synced(
                db,
                delta.existing_link,
                now=now,
                content_checksum=canonical.effective_checksum(),
                document_id=doc.id,
                document_version_id=getattr(doc, "version_id", None),
                external_uri=canonical.source_uri,
                remote_updated_at=canonical.updated_at,
            )
            result.updated_count += 1

        logger.debug(
            "source_sync.delta external_id=%s action=%s doc_id=%s reason=%s",
            external_id,
            delta.action.value,
            doc.id,
            delta.reason,
        )

    # ── Legacy item sync (no link tracking) ──────────────────────

    async def _sync_single_item_legacy(
        self,
        db: AsyncSession,
        *,
        connector: Any,
        tenant_id: str,
        source_key: str,
        external_id: str,
        ref: dict[str, Any],
        result: SyncResult,
    ) -> None:
        """Fetch, map, and upsert a single item (original behavior)."""

        # Fetch detail
        detail = await connector.fetch_item_detail(
            tenant_id=tenant_id,
            source_key=source_key,
            external_id=external_id,
            ref=ref,
        )
        if detail is None:
            result.unchanged_count += 1
            logger.debug(
                "source_sync.detail_none external_id=%s source_key=%s",
                external_id,
                source_key,
            )
            return

        # Map to canonical
        canonical = connector.map_to_canonical_item(
            source_key=source_key,
            raw_detail=detail,
        )
        if canonical is None:
            result.error_count += 1
            self._record_error(
                result, "map_failed", external_id,
                "map_to_canonical_item returned None",
            )
            logger.debug(
                "source_sync.map_none external_id=%s source_key=%s",
                external_id,
                source_key,
            )
            return

        # Upsert via DocumentService
        payload = self._build_upsert_payload(canonical, tenant_id=tenant_id)

        doc, action, _processed = await self._doc_svc.upsert(db, **payload)

        if action == "noop":
            result.unchanged_count += 1
        elif action == "created":
            result.created_count += 1
        else:
            result.updated_count += 1

        logger.debug(
            "source_sync.upserted external_id=%s doc_id=%s action=%s",
            external_id,
            doc.id,
            action,
        )

    # ── Payload builder ──────────────────────────────────────────

    @staticmethod
    def _build_upsert_payload(
        canonical: CanonicalKnowledgeItem,
        *,
        tenant_id: str,
    ) -> dict[str, Any]:
        """Build keyword arguments for ``DocumentService.upsert()``.

        Delegates to the shared ``build_upsert_payload()`` helper
        to ensure all source platform flows use identical ingest payloads.
        """
        from app.services.source_platform.runtime_mapping import (
            build_upsert_payload,
        )

        return build_upsert_payload(canonical, tenant_id=tenant_id)

    @staticmethod
    def _record_error(
        result: SyncResult,
        phase: str,
        external_id: str,
        message: str,
    ) -> None:
        """Append a sanitised error entry (capped to avoid bloat)."""
        if len(result.errors) < _MAX_ERROR_LOG_ITEMS:
            result.errors.append(
                {
                    "phase": phase,
                    "external_id": external_id,
                    "error": message,
                }
            )
