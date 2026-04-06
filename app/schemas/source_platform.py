"""
Source platform Pydantic schemas (Phase 4 + Phase 7).

Request/response models for the admin source platform API.
Auth secrets are **always** masked in responses.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ── Secret masking ───────────────────────────────────────────────────

_MASK = "****"


def mask_auth_config(auth_config: dict[str, Any] | None) -> dict[str, Any] | None:
    """Mask sensitive values in auth_config for API responses.

    Shows auth_type structure but replaces actual secrets with ``****``.
    Preserves last 4 chars of tokens for operator identification.
    """
    if not auth_config:
        return auth_config

    masked = {}
    for key, value in auth_config.items():
        if key in ("token", "api_key", "secret", "password"):
            if isinstance(value, str) and len(value) > 4:
                masked[key] = f"{_MASK}{value[-4:]}"
            else:
                masked[key] = _MASK
        else:
            masked[key] = value
    return masked


# ── Allowed values ───────────────────────────────────────────────────

ALLOWED_CONNECTOR_TYPES = frozenset({"internal-api"})
# Phase 4 supports bearer token auth and no-auth only.
# api_key as a distinct auth type is reserved for a future phase
# when the connector runtime can differentiate it from bearer.
ALLOWED_AUTH_TYPES = frozenset({"bearer", "none"})


# ── Request schemas ──────────────────────────────────────────────────


class SourceCreateRequest(BaseModel):
    """Create a new onboarded source."""

    source_key: str = Field(
        ..., min_length=1, max_length=128,
        description="Unique runtime identifier for this source",
    )
    name: str = Field(
        ..., min_length=1, max_length=256,
        description="Human-readable display name",
    )
    description: str | None = Field(
        None, max_length=2000,
    )
    connector_type: str = Field(
        ..., min_length=1, max_length=64,
        description="Connector type, e.g. 'internal-api'",
    )
    base_url: str = Field(
        ..., min_length=1, max_length=1024,
        description="Root URL of the source API",
    )
    auth_type: str = Field(
        "bearer", max_length=32,
        description="Authentication type: 'bearer' or 'none'",
    )
    auth_config: dict[str, Any] | None = Field(
        None,
        description='Auth secrets for bearer auth, e.g. {"token": "..."}',
    )
    list_path: str = Field(
        "/api/internal/knowledge/items",
        max_length=512,
    )
    detail_path_template: str | None = Field(
        None, max_length=512,
        description="Path template with {external_id} placeholder",
    )
    request_config: dict[str, Any] | None = Field(
        None,
        description=(
            "Reserved for future use. Stored but not applied to "
            "sync execution in the current phase."
        ),
    )
    mapping_config: dict[str, Any] | None = Field(
        None,
        description=(
            "Reserved for future use. Stored but not applied to "
            "sync execution in the current phase."
        ),
    )
    default_metadata: dict[str, Any] | None = None
    is_active: bool = True
    sync_enabled: bool = False
    sync_interval_minutes: int = Field(60, ge=1, le=10080)

    @field_validator("connector_type")
    @classmethod
    def validate_connector_type(cls, v: str) -> str:
        if v not in ALLOWED_CONNECTOR_TYPES:
            raise ValueError(
                f"connector_type must be one of: {sorted(ALLOWED_CONNECTOR_TYPES)}"
            )
        return v

    @field_validator("auth_type")
    @classmethod
    def validate_auth_type(cls, v: str) -> str:
        if v not in ALLOWED_AUTH_TYPES:
            raise ValueError(
                f"auth_type must be one of: {sorted(ALLOWED_AUTH_TYPES)}"
            )
        return v

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        return v.rstrip("/")


class SourceUpdateRequest(BaseModel):
    """Partial update for an onboarded source."""

    name: str | None = Field(None, min_length=1, max_length=256)
    description: str | None = None
    base_url: str | None = Field(None, min_length=1, max_length=1024)
    auth_type: str | None = Field(None, max_length=32)
    auth_config: dict[str, Any] | None = None
    list_path: str | None = Field(None, max_length=512)
    detail_path_template: str | None = None
    request_config: dict[str, Any] | None = Field(
        None,
        description="Reserved for future use. Stored but not applied to sync.",
    )
    mapping_config: dict[str, Any] | None = Field(
        None,
        description="Reserved for future use. Stored but not applied to sync.",
    )
    default_metadata: dict[str, Any] | None = None
    is_active: bool | None = None

    @field_validator("auth_type")
    @classmethod
    def validate_auth_type(cls, v: str | None) -> str | None:
        if v is not None and v not in ALLOWED_AUTH_TYPES:
            raise ValueError(
                f"auth_type must be one of: {sorted(ALLOWED_AUTH_TYPES)}"
            )
        return v

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, v: str | None) -> str | None:
        if v is not None:
            if not v.startswith(("http://", "https://")):
                raise ValueError("base_url must start with http:// or https://")
            return v.rstrip("/")
        return v


class SourceScheduleUpdateRequest(BaseModel):
    """Update sync schedule for a source (Phase 7)."""

    sync_enabled: bool | None = None
    sync_interval_minutes: int | None = Field(None, ge=1, le=10080)


# ── Response schemas ─────────────────────────────────────────────────


class SourceResponse(BaseModel):
    """Single source detail — auth_config is always masked."""

    id: int
    tenant_id: str
    source_key: str
    name: str
    description: str | None
    connector_type: str
    base_url: str
    auth_type: str
    auth_config: dict[str, Any] | None  # masked
    list_path: str
    detail_path_template: str | None
    request_config: dict[str, Any] | None  # reserved, not applied to sync yet
    mapping_config: dict[str, Any] | None  # reserved, not applied to sync yet
    default_metadata: dict[str, Any] | None
    is_active: bool
    last_synced_at: datetime | None
    # Phase 7 scheduling fields
    sync_enabled: bool = False
    sync_interval_minutes: int = 60
    next_sync_at: datetime | None = None
    last_sync_attempt_at: datetime | None = None
    # Phase 7 health summary
    health_status: str | None = None
    created_at: datetime | None
    updated_at: datetime | None


class SourceListResponse(BaseModel):
    """Paginated list of sources."""

    items: list[SourceResponse]
    total: int
    page: int
    page_size: int


class SyncTriggerResponse(BaseModel):
    """Response from triggering a sync."""

    sync_run_id: int
    source_id: int
    source_key: str
    status: str
    items_fetched: int = 0
    items_created: int = 0
    items_updated: int = 0
    items_unchanged: int = 0
    items_missing: int = 0
    items_reactivated: int = 0
    items_upserted: int = 0
    items_failed: int = 0
    message: str


class SyncRunResponse(BaseModel):
    """Single sync run detail."""

    id: int
    tenant_id: str
    source_id: int
    source_key: str
    status: str
    started_at: datetime | None
    finished_at: datetime | None
    items_fetched: int
    items_created: int = 0
    items_updated: int = 0
    items_unchanged: int = 0
    items_missing: int = 0
    items_reactivated: int = 0
    items_upserted: int
    items_failed: int
    error_message: str | None
    triggered_by: str | None
    created_at: datetime | None


class SyncRunListResponse(BaseModel):
    """List of sync runs for a source."""

    items: list[SyncRunResponse]
    total: int


class SourceHealthResponse(BaseModel):
    """Health status for a single source (Phase 7)."""

    source_id: int
    source_key: str
    is_active: bool
    sync_enabled: bool
    health_status: str
    consecutive_failures: int
    last_success_at: datetime | None
    last_failure_at: datetime | None
    last_sync_attempt_at: datetime | None
    last_error_message: str | None
    next_sync_at: datetime | None
    sync_interval_minutes: int


# ── Phase 2C: Link Repair schemas ────────────────────────────────────


class SourceRepairRequest(BaseModel):
    """Request body for repairing/reprocessing a single source link."""

    force_reprocess: bool = Field(
        False,
        description=(
            "If true, re-upserts the document even when content checksum "
            "is unchanged (overrides SKIP → force_reprocessed)."
        ),
    )


class SourceRepairResponse(BaseModel):
    """Operator-facing result of a link repair/reprocess operation (Phase 2C).

    Covers all outcomes: repaired, reactivated, updated, unchanged,
    force_reprocessed, or failed.
    """

    source_id: int
    source_key: str
    link_id: int
    external_id: str
    ok: bool
    action_taken: str = Field(
        description=(
            "One of: reactivated | updated | unchanged | "
            "force_reprocessed | failed"
        ),
    )
    previous_link_status: str | None
    new_link_status: str | None
    document_id: int | None = None
    document_version_id: str | None = None
    content_checksum: str | None = None
    had_document_before: bool = False
    raw_item_found: bool = False
    canonical_mapped: bool = False
    message: str
    warnings: list[str] = Field(default_factory=list)
    repaired_at: datetime


# ── Phase 1: Observability schemas ───────────────────────────────────


class SourceLinkCounts(BaseModel):
    """Aggregated link status counts for a source."""

    total: int = 0
    active: int = 0
    missing: int = 0
    error: int = 0


class SourceAttentionSummary(BaseModel):
    """Derived attention flags for operator dashboards."""

    needs_attention: bool = False
    is_stale: bool = False
    is_degraded: bool = False
    is_failing: bool = False


class SourceOverviewResponse(BaseModel):
    """Control-room overview for a single source."""

    source_id: int
    tenant_id: str
    source_key: str
    name: str
    description: str | None = None
    connector_type: str
    is_active: bool
    sync_enabled: bool
    sync_interval_minutes: int
    next_sync_at: datetime | None = None
    last_synced_at: datetime | None = None
    last_success_at: datetime | None = None
    last_failure_at: datetime | None = None
    consecutive_failures: int = 0
    last_error_message: str | None = None
    link_counts: SourceLinkCounts
    attention: SourceAttentionSummary
    operational_status: str
    created_at: datetime | None = None
    updated_at: datetime | None = None


class SourceLinkItemResponse(BaseModel):
    """Single link item in a paginated list."""

    id: int
    tenant_id: str
    onboarded_source_id: int
    external_id: str
    status: str
    document_id: int | None = None
    document_version_id: str | None = None
    content_checksum: str | None = None
    external_uri: str | None = None
    remote_updated_at: datetime | None = None
    last_seen_at: datetime | None = None
    last_synced_at: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class SourceLinkListResponse(BaseModel):
    """Paginated list of source document links."""

    items: list[SourceLinkItemResponse]
    total: int
    page: int
    page_size: int


class SourceLinkDetailResponse(BaseModel):
    """Detailed view of a single source document link."""

    id: int
    tenant_id: str
    onboarded_source_id: int
    source_key: str
    external_id: str
    status: str
    document_id: int | None = None
    document_version_id: str | None = None
    content_checksum: str | None = None
    external_uri: str | None = None
    remote_updated_at: datetime | None = None
    last_seen_at: datetime | None = None
    last_synced_at: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    metadata_json: dict[str, Any] | None = None
    has_document: bool = False
    is_missing: bool = False
    source_name: str
    source_connector_type: str


class SyncRunDetailResponse(BaseModel):
    """Detailed view of a single sync run with derived outcome."""

    id: int
    tenant_id: str
    source_id: int
    source_key: str
    status: str
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_seconds: float | None = None
    items_fetched: int = 0
    items_created: int = 0
    items_updated: int = 0
    items_unchanged: int = 0
    items_missing: int = 0
    items_reactivated: int = 0
    items_upserted: int = 0
    items_failed: int = 0
    error_message: str | None = None
    triggered_by: str | None = None
    created_at: datetime | None = None
    derived_outcome: str


# ── Phase 2A: Investigation schemas ──────────────────────────────────


class SourceConnectionTestResponse(BaseModel):
    """Result of a connectivity test to a source."""

    source_id: int
    source_key: str
    connector_type: str
    ok: bool
    tested_at: datetime
    latency_ms: float | None = None
    message: str
    error_type: str | None = None


class SourcePreviewRefItem(BaseModel):
    """Single item ref from a preview fetch."""

    external_id: str
    title: str | None = None
    updated_at: str | None = None
    kind: str | None = None
    uri: str | None = None


class SourcePreviewRefsResponse(BaseModel):
    """Preview of item refs from a source."""

    source_id: int
    source_key: str
    connector_type: str
    fetched_count: int = 0
    returned_count: int = 0
    items: list[SourcePreviewRefItem] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    tested_at: datetime


class CanonicalItemPreview(BaseModel):
    """Preview of a canonical-mapped item (read-only, no side effects)."""

    external_id: str
    title: str | None = None
    summary: str | None = None
    body_text_preview: str | None = None
    source_uri: str | None = None
    updated_at: datetime | None = None
    checksum: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    access_scope: dict[str, Any] = Field(default_factory=dict)


class PreviewValidationSummary(BaseModel):
    """Validation summary for a previewed item."""

    can_map: bool | None = None
    has_external_id: bool | None = None
    has_title: bool | None = None
    has_meaningful_body: bool | None = None
    body_length: int | None = None
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class SourceItemPreviewResponse(BaseModel):
    """Full preview of a single source item: raw, canonical, validation."""

    source_id: int
    source_key: str
    external_id: str
    tested_at: datetime
    raw_detail: dict[str, Any] | None = None
    canonical: CanonicalItemPreview | None = None
    validation: PreviewValidationSummary = Field(
        default_factory=PreviewValidationSummary
    )


# ── Phase 2B: Action schemas ────────────────────────────────────────


class SyncOneRequest(BaseModel):
    """Request body for syncing a single item by external_id."""

    external_id: str = Field(
        ..., min_length=1,
        description="External ID of the item to sync from the source",
    )
    force_reprocess: bool = Field(
        False,
        description=(
            "If true, re-upserts the document even when content checksum "
            "is unchanged (overrides SKIP → force_reprocessed)."
        ),
    )


class SyncOneResponse(BaseModel):
    """Operator-facing result of a single-item sync operation (Phase 2B)."""

    source_id: int
    source_key: str
    external_id: str
    ok: bool
    action_taken: str = Field(
        description=(
            "One of: created | updated | reactivated | unchanged | "
            "force_reprocessed | failed"
        ),
    )
    document_id: int | None = None
    document_version_id: str | None = None
    content_checksum: str | None = None
    had_existing_link: bool = False
    previous_link_status: str | None = None
    canonical_mapped: bool = False
    message: str
    warnings: list[str] = Field(default_factory=list)
    synced_at: datetime
