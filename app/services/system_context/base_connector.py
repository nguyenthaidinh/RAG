"""
Base system connector protocol (Phase 1 — Foundation).

Defines the read-only interface that every system connector must
implement.  Uses typing.Protocol so connectors don't need to inherit
from a concrete base class — duck typing is sufficient.

Fail semantics:
  - Permission-related methods should fail CLOSED (raise or return
    empty/deny-all snapshot).
  - Stats / record / workflow methods may fail OPEN (return None,
    log warning).

Phase 2 will add the real HTTP connector for core-platform.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.schemas.system_context import (
    PermissionSnapshot,
    RecordSummary,
    SystemContextBundle,
    TenantContext,
    TenantStats,
    UserContext,
    WorkflowSummary,
)


@runtime_checkable
class BaseSystemConnector(Protocol):
    """
    Read-only protocol for system context connectors.

    Every method receives explicit tenant_id and actor_user_id so
    connectors never have to infer identity from ambient state.
    """

    @property
    def provider_name(self) -> str:
        """Short identifier, e.g. 'mock', 'core-platform'."""
        ...

    async def get_user_context(
        self,
        *,
        tenant_id: str,
        actor_user_id: str | int,
    ) -> UserContext | None:
        """Fetch identity snapshot for the acting user."""
        ...

    async def get_tenant_context(
        self,
        *,
        tenant_id: str,
    ) -> TenantContext | None:
        """Fetch tenant identity + attributes."""
        ...

    async def get_permission_snapshot(
        self,
        *,
        tenant_id: str,
        actor_user_id: str | int,
        resource_types: list[str] | None = None,
    ) -> PermissionSnapshot | None:
        """
        Fetch permission decisions for the actor.

        Args:
            resource_types: Optional filter — only check these types.
                            None = check all available.

        Fail-closed: if the connector cannot determine permissions,
        it should return an empty snapshot (no decisions = deny-all).
        """
        ...

    async def get_tenant_stats(
        self,
        *,
        tenant_id: str,
        period: str | None = None,
    ) -> TenantStats | None:
        """Fetch aggregated tenant metrics.  Fail-open: may return None."""
        ...

    async def get_record_summaries(
        self,
        *,
        tenant_id: str,
        actor_user_id: str | int | None = None,
        record_types: list[str] | None = None,
        limit: int = 20,
    ) -> list[RecordSummary]:
        """Fetch lightweight record summaries.  Fail-open: may return []."""
        ...

    async def get_workflow_summaries(
        self,
        *,
        tenant_id: str,
        workflow_types: list[str] | None = None,
    ) -> list[WorkflowSummary]:
        """Fetch workflow status summaries.  Fail-open: may return []."""
        ...

    async def build_context_bundle(
        self,
        *,
        tenant_id: str,
        actor_user_id: str | int,
        include_user: bool = True,
        include_tenant: bool = True,
        include_permissions: bool = False,
        include_stats: bool = False,
        include_records: bool = False,
        include_workflows: bool = False,
    ) -> SystemContextBundle:
        """
        Convenience: assemble a full SystemContextBundle.

        Default implementation is in SystemContextBuilder.
        Connectors MAY override if they can batch-fetch more efficiently.
        """
        ...
