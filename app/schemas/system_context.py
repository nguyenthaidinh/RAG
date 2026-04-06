"""
System Context schemas (Phase 1.1 — Hardened Foundation).

Domain-neutral DTOs for system context data sourced from external
platforms (e.g. core-platform).  These schemas define the contract
between AI Server and any system connector.

Design principles:
  - Pydantic v2 BaseModel for validation + serialization
  - Safe defaults — every optional field has a sensible fallback
  - Tenant-sensitive fields are REQUIRED — never silently default
  - Domain-neutral — no business-specific field names
  - Read-only — no mutation verbs
  - Typed — all fields annotated
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────────────────────
# Actor / Identity
# ──────────────────────────────────────────────────────────────────────────────


class UserContext(BaseModel):
    """Identity + role snapshot of the acting user."""

    user_id: str | int
    email: str | None = None
    display_name: str | None = None
    tenant_id: str
    role: str | None = None
    roles: list[str] = Field(default_factory=list)
    scopes: list[str] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)
    is_active: bool = True


class TenantContext(BaseModel):
    """Tenant identity + attributes snapshot."""

    tenant_id: str
    tenant_name: str | None = None
    tenant_slug: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ──────────────────────────────────────────────────────────────────────────────
# Permissions
# ──────────────────────────────────────────────────────────────────────────────


class PermissionDecision(BaseModel):
    """A single permission check result."""

    resource_type: str
    action: str
    allowed: bool
    scope: str | None = None
    reason: str | None = None
    field_masking: dict[str, str] = Field(default_factory=dict)


class PermissionSnapshot(BaseModel):
    """Batch of permission decisions for a specific actor + tenant.

    Both tenant_id and actor_user_id are required — a permission
    snapshot without an actor is semantically invalid.
    """

    tenant_id: str
    actor_user_id: str | int
    decisions: list[PermissionDecision] = Field(default_factory=list)
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ──────────────────────────────────────────────────────────────────────────────
# Statistics / Metrics
# ──────────────────────────────────────────────────────────────────────────────


class MetricValue(BaseModel):
    """A single named metric value."""

    key: str
    value: int | float | str | bool
    label: str | None = None
    unit: str | None = None


class TenantStats(BaseModel):
    """Aggregated tenant-level statistics."""

    tenant_id: str
    metrics: list[MetricValue] = Field(default_factory=list)
    period: str | None = None
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ──────────────────────────────────────────────────────────────────────────────
# Records / Workflows
# ──────────────────────────────────────────────────────────────────────────────


class RecordSummary(BaseModel):
    """Lightweight summary of a business record.

    tenant_id is REQUIRED — records must always be tenant-scoped.
    """

    record_type: str
    record_id: str
    title: str | None = None
    status: str | None = None
    owner_id: str | None = None
    tenant_id: str
    summary: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class WorkflowSummary(BaseModel):
    """Aggregated workflow/process status summary.

    tenant_id is REQUIRED — workflow summaries must be tenant-scoped.
    """

    workflow_type: str
    tenant_id: str
    total: int | None = None
    by_status: dict[str, int] = Field(default_factory=dict)
    pending_count: int | None = None
    completed_count: int | None = None
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ──────────────────────────────────────────────────────────────────────────────
# Bundle
# ──────────────────────────────────────────────────────────────────────────────


class SystemContextBundle(BaseModel):
    """
    Complete system context snapshot for a single request.

    Assembled by SystemContextBuilder from connector responses.
    Consumer code should treat this as read-only.
    """

    user: UserContext | None = None
    tenant: TenantContext | None = None
    permissions: PermissionSnapshot | None = None
    tenant_stats: TenantStats | None = None
    records: list[RecordSummary] = Field(default_factory=list)
    workflows: list[WorkflowSummary] = Field(default_factory=list)
    source: str = "unknown"
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def has_user(self) -> bool:
        return self.user is not None

    @property
    def has_permissions(self) -> bool:
        return self.permissions is not None and len(self.permissions.decisions) > 0

    @property
    def has_stats(self) -> bool:
        return self.tenant_stats is not None and len(self.tenant_stats.metrics) > 0
