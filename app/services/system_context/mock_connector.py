"""
Mock system connector (Phase 1 — Foundation).

Returns deterministic, tenant-aware fake data for dev/test.
All data is structurally valid and tenant-scoped, allowing
the full context pipeline to be exercised without a real
external system.

NOT for production use — gated by connector registry selection.
"""
from __future__ import annotations

from datetime import datetime, timezone

from app.schemas.system_context import (
    MetricValue,
    PermissionDecision,
    PermissionSnapshot,
    RecordSummary,
    SystemContextBundle,
    TenantContext,
    TenantStats,
    UserContext,
    WorkflowSummary,
)


class MockSystemConnector:
    """Deterministic mock connector for dev/test."""

    @property
    def provider_name(self) -> str:
        return "mock"

    async def get_user_context(
        self,
        *,
        tenant_id: str,
        actor_user_id: str | int,
    ) -> UserContext:
        return UserContext(
            user_id=actor_user_id,
            email=f"user-{actor_user_id}@{tenant_id}.test",
            display_name=f"Test User {actor_user_id}",
            tenant_id=tenant_id,
            role="user",
            roles=["user"],
            scopes=["read"],
            attributes={"mock": True},
            is_active=True,
        )

    async def get_tenant_context(
        self,
        *,
        tenant_id: str,
    ) -> TenantContext:
        return TenantContext(
            tenant_id=tenant_id,
            tenant_name=f"Tenant {tenant_id}",
            tenant_slug=tenant_id.lower().replace(" ", "-"),
            attributes={"plan": "pro", "mock": True},
            fetched_at=datetime.now(timezone.utc),
        )

    async def get_permission_snapshot(
        self,
        *,
        tenant_id: str,
        actor_user_id: str | int,
        resource_types: list[str] | None = None,
    ) -> PermissionSnapshot:
        """Return a permissive snapshot for testing."""
        types = resource_types or ["document", "query", "report"]
        decisions = [
            PermissionDecision(
                resource_type=rt,
                action="read",
                allowed=True,
                scope="tenant",
                reason="mock_allow_all",
            )
            for rt in types
        ]
        return PermissionSnapshot(
            tenant_id=tenant_id,
            actor_user_id=actor_user_id,
            decisions=decisions,
            fetched_at=datetime.now(timezone.utc),
        )

    async def get_tenant_stats(
        self,
        *,
        tenant_id: str,
        period: str | None = None,
    ) -> TenantStats:
        return TenantStats(
            tenant_id=tenant_id,
            metrics=[
                MetricValue(key="total_users", value=42, label="Total Users"),
                MetricValue(key="active_users", value=38, label="Active Users"),
                MetricValue(key="documents_count", value=150, label="Documents"),
                MetricValue(key="queries_today", value=89, label="Queries Today"),
            ],
            period=period or "current",
            fetched_at=datetime.now(timezone.utc),
        )

    async def get_record_summaries(
        self,
        *,
        tenant_id: str,
        actor_user_id: str | int | None = None,
        record_types: list[str] | None = None,
        limit: int = 20,
    ) -> list[RecordSummary]:
        return [
            RecordSummary(
                record_type="request",
                record_id="REQ-001",
                title="Sample Request #1",
                status="pending",
                owner_id=str(actor_user_id) if actor_user_id else None,
                tenant_id=tenant_id,
                summary="A mock pending request for testing.",
                metadata={"priority": "normal", "mock": True},
                fetched_at=datetime.now(timezone.utc),
            ),
            RecordSummary(
                record_type="request",
                record_id="REQ-002",
                title="Sample Request #2",
                status="approved",
                owner_id=str(actor_user_id) if actor_user_id else None,
                tenant_id=tenant_id,
                summary="A mock approved request.",
                metadata={"priority": "high", "mock": True},
                fetched_at=datetime.now(timezone.utc),
            ),
        ][:limit]

    async def get_workflow_summaries(
        self,
        *,
        tenant_id: str,
        workflow_types: list[str] | None = None,
    ) -> list[WorkflowSummary]:
        return [
            WorkflowSummary(
                workflow_type="approval",
                tenant_id=tenant_id,
                total=25,
                by_status={"pending": 5, "approved": 18, "rejected": 2},
                pending_count=5,
                completed_count=20,
                fetched_at=datetime.now(timezone.utc),
            ),
        ]

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
        """Assemble a full bundle from individual mock methods."""
        now = datetime.now(timezone.utc)

        user = (
            await self.get_user_context(
                tenant_id=tenant_id, actor_user_id=actor_user_id,
            )
            if include_user
            else None
        )
        tenant = (
            await self.get_tenant_context(tenant_id=tenant_id)
            if include_tenant
            else None
        )
        permissions = (
            await self.get_permission_snapshot(
                tenant_id=tenant_id, actor_user_id=actor_user_id,
            )
            if include_permissions
            else None
        )
        stats = (
            await self.get_tenant_stats(tenant_id=tenant_id)
            if include_stats
            else None
        )
        records = (
            await self.get_record_summaries(
                tenant_id=tenant_id, actor_user_id=actor_user_id,
            )
            if include_records
            else []
        )
        workflows = (
            await self.get_workflow_summaries(tenant_id=tenant_id)
            if include_workflows
            else []
        )

        return SystemContextBundle(
            user=user,
            tenant=tenant,
            permissions=permissions,
            tenant_stats=stats,
            records=records,
            workflows=workflows,
            source=self.provider_name,
            fetched_at=now,
        )
