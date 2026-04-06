"""Shared runtime mapping for onboarded sources.

Single source of truth for:
  - mapping an ``OnboardedSource`` DB row to ``OnboardedSourceConfig``
  - building ``DocumentService.upsert()`` kwargs from a canonical item

All services in the source platform package (admin, investigation,
action, repair, sync) MUST use these helpers instead of duplicating
mapping logic.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.db.models.onboarded_source import OnboardedSource
from app.services.source_platform.canonical_item import CanonicalKnowledgeItem
from app.services.source_platform.onboarded_source_config import (
    OnboardedSourceConfig,
)


def row_to_runtime_config(row: OnboardedSource) -> OnboardedSourceConfig:
    """Map an ``OnboardedSource`` DB row to runtime config.

    Extracts auth token from ``auth_config`` JSON (``token`` or
    ``api_key`` key, whichever is present).
    Maps ``connector_type`` → ``connector_name``.
    Maps ``default_metadata`` → ``default_params``, popping
    ``content_kind`` out as a first-class config field.
    Falls back ``detail_path_template`` to the internal-API default
    when the DB value is empty/null.
    """
    # Extract auth token
    auth_token = ""
    if row.auth_config and isinstance(row.auth_config, dict):
        auth_token = (
            row.auth_config.get("token")
            or row.auth_config.get("api_key")
            or ""
        )

    # Extract content_kind from default_metadata
    default_params = dict(row.default_metadata or {})
    content_kind = default_params.pop("content_kind", "")

    return OnboardedSourceConfig(
        source_key=row.source_key,
        connector_name=row.connector_type,
        base_url=row.base_url,
        auth_token=auth_token,
        list_path=row.list_path,
        detail_path_template=(
            row.detail_path_template
            or "/api/internal/knowledge/items/{external_id}"
        ),
        content_kind=content_kind,
        default_params=default_params,
        enabled=row.is_active,
    )


def build_upsert_payload(
    canonical: CanonicalKnowledgeItem,
    *,
    tenant_id: str,
) -> dict[str, Any]:
    """Build keyword arguments for ``DocumentService.upsert()``.

    This is the **single source of truth** for building the ingest
    payload from a canonical knowledge item.  All source platform
    flows (full sync, sync-one, repair) MUST use this helper.

    Merges the canonical item's metadata with a ``source_platform``
    provenance block so downstream consumers can trace the origin.

    The ``source_platform`` block contains:
      - ``source_key``   — logical source identifier
      - ``source_type``  — connector type (e.g. ``internal_api``)
      - ``source_uri``   — link back to the source item
      - ``access_scope`` — access-control hints from upstream
      - ``kind``         — domain hint (``policy``, ``faq``, etc.)
      - ``synced_at``    — ISO timestamp of this sync execution
      - ``ingest_mode``  — always ``source_platform`` for platform-ingested docs
      - ``updated_at``   — source-side last-modified timestamp
      - ``checksum``     — content checksum for change detection
    """
    merged_metadata = {
        **canonical.metadata,
        "source_platform": {
            "source_key": canonical.source_key,
            "source_type": canonical.source_type,
            "source_uri": canonical.source_uri,
            "access_scope": canonical.access_scope,
            "kind": canonical.metadata.get("kind"),
            "synced_at": datetime.now(timezone.utc).isoformat(),
            "ingest_mode": "source_platform",
            "updated_at": (
                canonical.updated_at.isoformat()
                if canonical.updated_at
                else None
            ),
            "checksum": canonical.effective_checksum(),
        },
    }

    return {
        "tenant_id": tenant_id,
        "source": canonical.source_key,
        "external_id": canonical.external_id,
        "content": canonical.normalized_body_text(),
        "title": canonical.normalized_title(),
        "metadata": merged_metadata,
        "representation_type": canonical.representation_type,
    }

