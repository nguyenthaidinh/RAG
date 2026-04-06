"""
Onboarded source config (Phase 3 — Multi-Source Onboarding).

Lightweight runtime config that describes *one* onboarded knowledge source
without hard-coding any web-specific logic.

Each source/web that the AI Server ingests from is represented as an
``OnboardedSourceConfig`` instance.  The config carries enough information
to build a connector, register it, and run a sync — all via the generic
``sync_onboarded_source()`` flow.

Adding a second or third source requires **only** creating another config
instance — no new modules, no new service functions.

Example::

    config = OnboardedSourceConfig(
        source_key="acme-policies",
        connector_name="internal-api",
        base_url="https://api.acme.com",
        auth_token="secret",
        list_path="/api/internal/knowledge/items",
        detail_path_template="/api/internal/knowledge/items/{external_id}",
        content_kind="policy",
    )
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OnboardedSourceConfig:
    """Immutable description of a single onboarded knowledge source.

    Fields
    ------
    source_key : str
        Logical identifier for this source.  Maps to
        ``DocumentService.upsert(source=...)``.
        Must be unique across all onboarded sources.
    connector_name : str
        Connector type key used for ``SourceRegistry`` registration.
        For internal API sources this is typically ``"internal-api"``.
    base_url : str
        Root URL of the source's API (e.g. ``"https://api.acme.com"``).
    auth_token : str
        Bearer token / API key for the source API.
        Empty string means no auth header is sent.
    list_path : str
        API path for listing items
        (e.g. ``"/api/internal/knowledge/items"``).
    detail_path_template : str
        f-string template for fetching a single item detail.
        Must contain ``{external_id}`` placeholder.
        (e.g. ``"/api/internal/knowledge/items/{external_id}"``).
    content_kind : str
        Domain hint passed as ``kind`` param to the list endpoint.
        E.g. ``"policy"``, ``"faq"``, ``"article"``.
        Empty string means no kind filter.
    default_params : dict
        Extra query parameters merged into every list request.
        Useful for status filters, category filters, etc.
    enabled : bool
        Toggle.  Disabled configs are skipped by the sync flow.
    """

    source_key: str
    connector_name: str
    base_url: str
    auth_token: str = ""
    list_path: str = "/api/internal/knowledge/items"
    detail_path_template: str = "/api/internal/knowledge/items/{external_id}"
    content_kind: str = ""
    default_params: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    # ── Factory ──────────────────────────────────────────────────

    def build_connector(self) -> Any:
        """Build a connector instance from this config.

        Currently supports ``connector_name="internal-api"`` which
        produces an ``InternalApiConnector``.

        Raises ``ValueError`` for unsupported connector names so
        callers get a fast, clear error at onboarding time.
        """
        if self.connector_name == "internal-api":
            return self._build_internal_api_connector()

        raise ValueError(
            f"Unsupported connector_name='{self.connector_name}' "
            f"for source_key='{self.source_key}'.  "
            f"Supported: ['internal-api']"
        )

    def _build_internal_api_connector(self) -> Any:
        """Build an InternalApiConnector from config fields."""
        from app.services.source_platform.connectors.internal_api_connector import (
            InternalApiConnector,
        )

        connector = InternalApiConnector(
            base_url=self.base_url,
            api_key=self.auth_token or None,
            list_path=self.list_path,
            detail_path_template=self.detail_path_template,
        )

        logger.info(
            "source_platform.config.build_connector "
            "source_key=%s connector=%s base_url=%s",
            self.source_key,
            self.connector_name,
            self.base_url,
        )

        return connector

    def build_sync_params(
        self,
        extra_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build merged params dict for ``SourceSyncService.sync_items()``.

        Merges ``default_params`` + ``content_kind`` (if set) + any
        caller-provided ``extra_params``.  Caller params win on conflict.
        """
        params: dict[str, Any] = dict(self.default_params)

        if self.content_kind:
            params.setdefault("kind", self.content_kind)

        if extra_params:
            params.update(extra_params)

        return params
