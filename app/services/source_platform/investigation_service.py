"""Source platform investigation service (Phase 2A).

Read-only preview/investigation service for operator admin workflows.
Builds connectors from source config and exercises them without any
side effects — no DB writes, no sync runs, no document mutations.

Does NOT:
  - Create or update source_document_links
  - Create sync runs
  - Upsert documents
  - Update source health/schedule
  - Touch the scheduler or coordinator
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.onboarded_source import OnboardedSource
from app.repos.source_platform_repo import SourcePlatformRepo
from app.schemas.source_platform import (
    CanonicalItemPreview,
    PreviewValidationSummary,
    SourceConnectionTestResponse,
    SourceItemPreviewResponse,
    SourcePreviewRefItem,
    SourcePreviewRefsResponse,
)
from app.services.source_platform.runtime_mapping import (
    row_to_runtime_config,
)

logger = logging.getLogger(__name__)

# Body preview truncation limit (chars).
_BODY_PREVIEW_MAX = 1000


class SourcePlatformInvestigationService:
    """Preview/investigation service for admin source platform."""

    def __init__(
        self,
        source_repo: SourcePlatformRepo | None = None,
    ) -> None:
        self._source_repo = source_repo or SourcePlatformRepo()

    # ── Test Connection ──────────────────────────────────────────

    async def test_connection(
        self,
        db: AsyncSession,
        *,
        tenant_id: str,
        source_id: int,
    ) -> SourceConnectionTestResponse | None:
        """Test connectivity to a source via the production sync path.

        Uses ``fetch_item_refs(limit=1)`` instead of the connector's
        bare ``test_connection()`` so the test exercises the same
        tenant-scoped, param-merged HTTP path that a real sync would
        use.  Still zero side effects — ``fetch_item_refs`` is read-only.
        """
        source = await self._source_repo.get_source(
            db, tenant_id=tenant_id, source_id=source_id
        )
        if not source:
            return None

        config = self._row_to_config(source)
        now = datetime.now(timezone.utc)

        try:
            connector = config.build_connector()
        except ValueError as exc:
            return SourceConnectionTestResponse(
                source_id=source.id,
                source_key=source.source_key,
                connector_type=source.connector_type,
                ok=False,
                tested_at=now,
                message=f"Cannot build connector: {exc}",
                error_type="config_error",
            )

        # Use fetch_item_refs(limit=1) — same path as production sync,
        # including tenant_id in query params, X-Tenant-Id header,
        # content_kind, and default_params from source config.
        t0 = time.monotonic()
        try:
            params = config.build_sync_params({"limit": "1"})

            refs = await connector.fetch_item_refs(
                tenant_id=tenant_id,
                source_key=source.source_key,
                params=params,
            )
            elapsed_ms = (time.monotonic() - t0) * 1000

            return SourceConnectionTestResponse(
                source_id=source.id,
                source_key=source.source_key,
                connector_type=source.connector_type,
                ok=True,
                tested_at=now,
                latency_ms=round(elapsed_ms, 1),
                message=(
                    f"Connection successful — source returned "
                    f"{len(refs)} ref(s) for limit=1 probe"
                ),
            )

        except Exception as exc:
            elapsed_ms = (time.monotonic() - t0) * 1000
            error_type = self._classify_error(exc)

            logger.warning(
                "source_platform.investigation.test_connection "
                "error source_id=%d: %s",
                source.id,
                str(exc),
                exc_info=True,
            )
            return SourceConnectionTestResponse(
                source_id=source.id,
                source_key=source.source_key,
                connector_type=source.connector_type,
                ok=False,
                tested_at=now,
                latency_ms=round(elapsed_ms, 1),
                message=f"Connection test failed: {str(exc)[:500]}",
                error_type=error_type,
            )

    # ── Preview Refs ─────────────────────────────────────────────

    async def preview_refs(
        self,
        db: AsyncSession,
        *,
        tenant_id: str,
        source_id: int,
        limit: int = 10,
    ) -> SourcePreviewRefsResponse | None:
        """Preview item refs from a source.  Zero side effects."""
        source = await self._source_repo.get_source(
            db, tenant_id=tenant_id, source_id=source_id
        )
        if not source:
            return None

        config = self._row_to_config(source)
        now = datetime.now(timezone.utc)
        notes: list[str] = []

        try:
            connector = config.build_connector()
        except ValueError as exc:
            return SourcePreviewRefsResponse(
                source_id=source.id,
                source_key=source.source_key,
                connector_type=source.connector_type,
                fetched_count=0,
                returned_count=0,
                items=[],
                notes=[f"Cannot build connector: {exc}"],
                tested_at=now,
            )

        try:
            # Build params with limit, reusing config's default params
            params = config.build_sync_params({"limit": str(limit)})

            refs = await connector.fetch_item_refs(
                tenant_id=tenant_id,
                source_key=source.source_key,
                params=params,
            )
        except Exception as exc:
            logger.warning(
                "source_platform.investigation.preview_refs "
                "error source_id=%d: %s",
                source.id,
                str(exc),
                exc_info=True,
            )
            return SourcePreviewRefsResponse(
                source_id=source.id,
                source_key=source.source_key,
                connector_type=source.connector_type,
                fetched_count=0,
                returned_count=0,
                items=[],
                notes=[f"Fetch failed: {str(exc)[:500]}"],
                tested_at=now,
            )

        fetched_count = len(refs)

        # Respect limit
        truncated = refs[:limit]
        if fetched_count > limit:
            notes.append(
                f"Source returned {fetched_count} refs, "
                f"showing first {limit}"
            )

        items = [
            SourcePreviewRefItem(
                external_id=r.get("external_id", ""),
                title=r.get("title"),
                updated_at=str(r["updated_at"]) if r.get("updated_at") else None,
                kind=r.get("kind"),
                uri=r.get("uri"),
            )
            for r in truncated
        ]

        return SourcePreviewRefsResponse(
            source_id=source.id,
            source_key=source.source_key,
            connector_type=source.connector_type,
            fetched_count=fetched_count,
            returned_count=len(items),
            items=items,
            notes=notes,
            tested_at=now,
        )

    # ── Preview Item Detail ──────────────────────────────────────

    async def preview_item_detail(
        self,
        db: AsyncSession,
        *,
        tenant_id: str,
        source_id: int,
        external_id: str,
    ) -> SourceItemPreviewResponse | None:
        """Preview a single item with raw, canonical, and validation layers.

        Returns ``None`` only if source itself is not found (404).
        If the item is not found at the source, returns a response
        with ``raw_detail=None`` and appropriate validation errors.
        """
        source = await self._source_repo.get_source(
            db, tenant_id=tenant_id, source_id=source_id
        )
        if not source:
            return None

        config = self._row_to_config(source)
        now = datetime.now(timezone.utc)

        try:
            connector = config.build_connector()
        except ValueError as exc:
            return SourceItemPreviewResponse(
                source_id=source.id,
                source_key=source.source_key,
                external_id=external_id,
                tested_at=now,
                validation=PreviewValidationSummary(
                    errors=[f"Cannot build connector: {exc}"],
                ),
            )

        # Fetch raw detail
        try:
            raw_detail = await connector.fetch_item_detail(
                tenant_id=tenant_id,
                source_key=source.source_key,
                external_id=external_id,
            )
        except Exception as exc:
            logger.warning(
                "source_platform.investigation.preview_item "
                "fetch_error source_id=%d external_id=%s: %s",
                source.id,
                external_id,
                str(exc),
                exc_info=True,
            )
            return SourceItemPreviewResponse(
                source_id=source.id,
                source_key=source.source_key,
                external_id=external_id,
                tested_at=now,
                validation=PreviewValidationSummary(
                    errors=[f"Fetch failed: {str(exc)[:500]}"],
                ),
            )

        if raw_detail is None:
            return SourceItemPreviewResponse(
                source_id=source.id,
                source_key=source.source_key,
                external_id=external_id,
                tested_at=now,
                validation=PreviewValidationSummary(
                    errors=[
                        f"Item not found at source for "
                        f"external_id='{external_id}'"
                    ],
                ),
            )

        # Attempt canonical mapping (pure function, no side effects)
        canonical_preview = None
        validation = self._build_validation(raw_detail)

        try:
            canonical = connector.map_to_canonical_item(
                source_key=source.source_key,
                raw_detail=raw_detail,
            )
            if canonical:
                validation.can_map = True
                validation.has_external_id = True
                validation.has_title = bool(canonical.title)
                body = canonical.normalized_body_text()
                validation.has_meaningful_body = len(body) >= 10
                validation.body_length = len(body)

                # Truncate body for preview
                body_preview = body[:_BODY_PREVIEW_MAX]
                if len(body) > _BODY_PREVIEW_MAX:
                    body_preview += "…"
                    validation.warnings.append(
                        f"Body truncated for preview "
                        f"({len(body)} chars, showing {_BODY_PREVIEW_MAX})"
                    )

                if not validation.has_title:
                    validation.warnings.append("No title found")

                canonical_preview = CanonicalItemPreview(
                    external_id=canonical.external_id,
                    title=canonical.normalized_title(),
                    summary=canonical.summary,
                    body_text_preview=body_preview,
                    source_uri=canonical.source_uri,
                    updated_at=canonical.updated_at,
                    checksum=canonical.effective_checksum(),
                    metadata=canonical.metadata,
                    access_scope=canonical.access_scope,
                )
            else:
                validation.can_map = False
                validation.errors.append(
                    "Canonical mapping returned None — item lacks "
                    "meaningful textual content or external_id"
                )
        except Exception as exc:
            validation.can_map = False
            validation.errors.append(
                f"Canonical mapping error: {str(exc)[:500]}"
            )

        return SourceItemPreviewResponse(
            source_id=source.id,
            source_key=source.source_key,
            external_id=external_id,
            tested_at=now,
            raw_detail=raw_detail,
            canonical=canonical_preview,
            validation=validation,
        )

    # ── Helpers ──────────────────────────────────────────────────

    @staticmethod
    def _row_to_config(row: OnboardedSource) -> OnboardedSourceConfig:
        """Map a DB row to runtime config.

        Delegates to the shared ``row_to_runtime_config()`` helper
        to ensure all source platform services use identical mapping.
        """
        return row_to_runtime_config(row)

    @staticmethod
    def _classify_error(exc: Exception) -> str:
        """Map an exception to an operator-friendly error_type label."""
        import httpx

        if isinstance(exc, httpx.TimeoutException):
            return "timeout"
        if isinstance(exc, httpx.ConnectError):
            return "connect_error"
        if isinstance(exc, httpx.HTTPStatusError):
            code = exc.response.status_code
            if code == 401 or code == 403:
                return "auth_error"
            return f"http_{code}"
        # Connector-internal errors (no base URL, oversized, etc.)
        cls_name = type(exc).__name__
        if "NoBaseURL" in cls_name:
            return "no_base_url"
        if "Oversized" in cls_name:
            return "oversized_response"
        return "unexpected_error"

    @staticmethod
    def _build_validation(raw_detail: dict[str, Any]) -> PreviewValidationSummary:
        """Build initial validation summary from raw detail payload."""
        v = PreviewValidationSummary()

        # Check external_id
        ext_id = (
            raw_detail.get("external_id")
            or raw_detail.get("id")
            or raw_detail.get("item_id")
            or raw_detail.get("slug")
        )
        v.has_external_id = bool(ext_id)
        if not v.has_external_id:
            v.errors.append(
                "No external_id found (checked: external_id, id, "
                "item_id, slug)"
            )

        # Check title
        title = (
            raw_detail.get("title")
            or raw_detail.get("name")
            or raw_detail.get("heading")
        )
        v.has_title = bool(title)

        return v
