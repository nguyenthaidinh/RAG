"""
Source onboarding service (Phase 3 — Multi-Source Onboarding).

Thin orchestration layer that takes an ``OnboardedSourceConfig`` and
runs a full sync cycle through the existing pipeline::

    OnboardedSourceConfig
      → build_connector()
      → SourceRegistry.register()
      → SourceSyncService.sync_items()
      → DocumentService.upsert()  (per item)

This module is **source-agnostic**.  It knows nothing about any
particular web or business domain.  All source-specific details live
in the ``OnboardedSourceConfig``.

Usage::

    from app.services.source_platform.source_onboarding_service import (
        sync_onboarded_source,
    )

    result = await sync_onboarded_source(
        db=session,
        config=my_config,
        tenant_id="tenant-123",
    )
    print(result.upserted_count)
"""
from __future__ import annotations

import logging
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.source_platform.onboarded_source_config import (
    OnboardedSourceConfig,
)
from app.services.source_platform.source_registry import get_source_registry
from app.services.source_platform.source_sync_service import (
    SourceSyncService,
    SyncResult,
)

logger = logging.getLogger(__name__)


# ── Public API ───────────────────────────────────────────────────────


async def sync_onboarded_source(
    db: AsyncSession,
    *,
    config: OnboardedSourceConfig,
    tenant_id: str,
    document_service: Any | None = None,
    params: dict[str, Any] | None = None,
    onboarded_source_id: int | None = None,
) -> SyncResult:
    """Run a full sync cycle for one onboarded source.

    This is the single entry-point for Phase 3 onboarding.  It:

    1. Validates the config is enabled.
    2. Builds a connector from the config.
    3. Ensures the connector is registered in ``SourceRegistry``.
    4. Builds merged params (config defaults + caller overrides).
    5. Delegates to ``SourceSyncService.sync_items()``.
    6. Returns ``SyncResult``.

    Parameters
    ----------
    db : AsyncSession
        Async DB session (caller owns the transaction).
    config : OnboardedSourceConfig
        The source to sync.
    tenant_id : str
        Tenant scope.
    document_service : DocumentService | None
        If ``None``, the global ``DocumentService`` singleton is used.
    params : dict | None
        Extra params merged on top of config defaults.
        Caller params win on key conflict.

    Returns
    -------
    SyncResult
        Counters and error summaries.

    Raises
    ------
    ValueError
        If the config is disabled or has an unsupported connector name.
    """
    if not config.enabled:
        logger.warning(
            "source_platform.onboarding.skip_disabled source_key=%s",
            config.source_key,
        )
        return SyncResult()

    logger.info(
        "source_platform.onboarding.start "
        "source_key=%s connector=%s tenant=%s",
        config.source_key,
        config.connector_name,
        tenant_id,
    )

    # ── 1. Build connector from config ───────────────────────────
    connector = config.build_connector()

    # ── 2. Register in SourceRegistry (idempotent) ───────────────
    registry = get_source_registry()
    _ensure_registered(registry, config.source_key, connector)

    # ── 3. Resolve DocumentService ───────────────────────────────
    doc_svc = document_service or _get_document_service()

    # ── 4. Build sync service + merged params ────────────────────
    sync_svc = SourceSyncService(registry=registry, document_service=doc_svc)
    merged_params = config.build_sync_params(params)

    # ── 5. Delegate to sync pipeline ─────────────────────────────
    result = await sync_svc.sync_items(
        db,
        tenant_id=tenant_id,
        provider_name=config.source_key,
        source_key=config.source_key,
        params=merged_params,
        onboarded_source_id=onboarded_source_id,
    )

    logger.info(
        "source_platform.onboarding.done "
        "source_key=%s tenant=%s "
        "discovered=%d created=%d updated=%d unchanged=%d "
        "missing=%d errors=%d",
        config.source_key,
        tenant_id,
        result.discovered_count,
        result.created_count,
        result.updated_count,
        result.unchanged_count,
        result.missing_count,
        result.error_count,
    )

    return result


# ── Helpers ──────────────────────────────────────────────────────────


def _ensure_registered(
    registry: Any,
    source_key: str,
    connector: Any,
) -> None:
    """Register (or replace) the connector under ``source_key``.

    Always overwrites so the connector built from the **current** config
    is authoritative.  This ensures that if ``base_url``, ``auth_token``,
    or endpoint paths change between calls, the registry reflects the
    latest config.

    Uses ``source_key`` (not ``connector_name``) as the registry key
    so multiple sources using the same connector type can co-exist
    (e.g. two different webs both using ``internal-api``).
    """
    is_update = source_key in registry
    registry.register(source_key, connector)

    if is_update:
        logger.info(
            "source_platform.onboarding.updated source_key=%s",
            source_key,
        )
    else:
        logger.info(
            "source_platform.onboarding.registered source_key=%s",
            source_key,
        )


def _get_document_service() -> Any:
    """Lazy-resolve the global DocumentService singleton."""
    from app.services.document_service import DocumentService

    return DocumentService()
