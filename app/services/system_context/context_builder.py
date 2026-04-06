"""
System context builder (Phase 1.1 — Hardened).

Assembles a SystemContextBundle by calling connector methods with
granular include flags.  Handles partial failures defensively:

  - Permission data: fail-closed (missing = empty snapshot → deny-all)
  - Stats / records / workflows: fail-open (missing = empty, log warning)
  - User / tenant: fail-open (missing = None, log warning)

The builder does NOT own identity resolution — tenant_id and
actor_user_id are always passed in from the authenticated context.

Phase 1.1 hardening:
  - Connector is type-annotated via BaseSystemConnector Protocol
  - Docstrings document fail semantics per section

Usage::

    from app.services.system_context.context_builder import SystemContextBuilder

    builder = SystemContextBuilder(connector=some_connector)
    bundle = await builder.build(
        tenant_id="t1",
        actor_user_id=42,
        flags=ContextBuildFlags(include_stats=True),
    )
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from dataclasses import dataclass

from app.schemas.system_context import (
    PermissionSnapshot,
    SystemContextBundle,
)
from app.services.system_context.base_connector import BaseSystemConnector

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContextBuildFlags:
    """Granular flags controlling which context parts to fetch.

    Used by both SystemContextBuilder and SystemContextOrchestrator
    to determine what data to request from the connector.

    Default: user + tenant only (minimal, safe).
    """

    include_user: bool = True
    include_tenant: bool = True
    include_permissions: bool = False
    include_stats: bool = False
    include_records: bool = False
    include_workflows: bool = False

    @property
    def needs_context_build(self) -> bool:
        """True if ANY flag is set that requires calling the connector."""
        return (
            self.include_user
            or self.include_tenant
            or self.include_permissions
            or self.include_stats
            or self.include_records
            or self.include_workflows
        )


class SystemContextBuilder:
    """
    Builds a SystemContextBundle from a connector.

    Orchestrates individual connector calls with defensive error handling.
    Each section is independently failable — a stats failure won't
    prevent user context from being returned.

    Fail semantics:
      - permissions: FAIL-CLOSED → empty PermissionSnapshot (no decisions = deny-all)
      - stats/records/workflows: fail-open → None/empty list
      - user/tenant: fail-open → None

    # TODO Phase 2+: add strict_access_mode parameter. When True,
    #   permission failure raises instead of returning empty snapshot.
    #   This is intentionally deferred until access context is wired
    #   into the production pipeline.
    """

    __slots__ = ("_connector",)

    def __init__(self, connector: BaseSystemConnector) -> None:
        """
        Args:
            connector: Object implementing BaseSystemConnector protocol.
        """
        self._connector = connector

    @property
    def provider_name(self) -> str:
        return getattr(self._connector, "provider_name", "unknown")

    async def build(
        self,
        *,
        tenant_id: str,
        actor_user_id: str | int,
        flags: ContextBuildFlags | None = None,
    ) -> SystemContextBundle:
        """
        Build a SystemContextBundle with the given flags.

        Args:
            tenant_id: Authenticated tenant (from JWT / API key).
            actor_user_id: Authenticated user ID.
            flags: Which parts to include.  Defaults to user + tenant only.

        Returns:
            SystemContextBundle — always returns a valid bundle,
            even if individual parts failed.
        """
        f = flags or ContextBuildFlags()
        now = datetime.now(timezone.utc)
        provider = self.provider_name

        user = None
        tenant = None
        permissions = None
        stats = None
        records = []
        workflows = []

        # ── User context (fail-open) ──────────────────────────────────
        if f.include_user:
            try:
                user = await self._connector.get_user_context(
                    tenant_id=tenant_id,
                    actor_user_id=actor_user_id,
                )
            except Exception:
                logger.warning(
                    "context_builder.user_failed provider=%s tenant_id=%s",
                    provider, tenant_id, exc_info=True,
                )

        # ── Tenant context (fail-open) ────────────────────────────────
        if f.include_tenant:
            try:
                tenant = await self._connector.get_tenant_context(
                    tenant_id=tenant_id,
                )
            except Exception:
                logger.warning(
                    "context_builder.tenant_failed provider=%s tenant_id=%s",
                    provider, tenant_id, exc_info=True,
                )

        # ── Permissions (FAIL-CLOSED: empty snapshot = deny-all) ──────
        if f.include_permissions:
            try:
                permissions = await self._connector.get_permission_snapshot(
                    tenant_id=tenant_id,
                    actor_user_id=actor_user_id,
                )
            except Exception:
                logger.warning(
                    "context_builder.permissions_failed provider=%s tenant_id=%s "
                    "behavior=FAIL_CLOSED returning_empty_snapshot",
                    provider, tenant_id, exc_info=True,
                )
                # Fail-closed: return empty snapshot (no decisions = no access)
                permissions = PermissionSnapshot(
                    tenant_id=tenant_id,
                    actor_user_id=actor_user_id,
                    fetched_at=now,
                )

        # ── Tenant stats (fail-open) ──────────────────────────────────
        if f.include_stats:
            try:
                stats = await self._connector.get_tenant_stats(
                    tenant_id=tenant_id,
                )
            except Exception:
                logger.warning(
                    "context_builder.stats_failed provider=%s tenant_id=%s",
                    provider, tenant_id, exc_info=True,
                )

        # ── Record summaries (fail-open) ──────────────────────────────
        if f.include_records:
            try:
                records = await self._connector.get_record_summaries(
                    tenant_id=tenant_id,
                    actor_user_id=actor_user_id,
                )
            except Exception:
                logger.warning(
                    "context_builder.records_failed provider=%s tenant_id=%s",
                    provider, tenant_id, exc_info=True,
                )
                records = []

        # ── Workflow summaries (fail-open) ─────────────────────────────
        if f.include_workflows:
            try:
                workflows = await self._connector.get_workflow_summaries(
                    tenant_id=tenant_id,
                )
            except Exception:
                logger.warning(
                    "context_builder.workflows_failed provider=%s tenant_id=%s",
                    provider, tenant_id, exc_info=True,
                )
                workflows = []

        bundle = SystemContextBundle(
            user=user,
            tenant=tenant,
            permissions=permissions,
            tenant_stats=stats,
            records=records,
            workflows=workflows,
            source=provider,
            fetched_at=now,
        )

        logger.info(
            "context_builder.done provider=%s tenant_id=%s "
            "has_user=%s has_tenant=%s has_permissions=%s "
            "has_stats=%s records=%d workflows=%d",
            provider, tenant_id,
            bundle.has_user, tenant is not None,
            bundle.has_permissions,
            bundle.has_stats,
            len(records), len(workflows),
        )

        return bundle
