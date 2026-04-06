"""
Base source connector protocol (Phase 1 — Foundation).

Defines the interface that every source connector must implement.
Uses ``typing.Protocol`` so connectors rely on structural subtyping
(duck typing) rather than class inheritance.

Responsibilities of a connector:
  1. Test connectivity to the upstream system.
  2. Fetch a list of item references (lightweight keys/ids).
  3. Fetch full detail for a single item.
  4. Map raw detail into a ``CanonicalKnowledgeItem``.

Connectors must NOT:
  - Call ``DocumentService.upsert()`` directly.
  - Manage job state or scheduling.
  - Perform side-effects beyond reading from their source.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from app.services.source_platform.canonical_item import CanonicalKnowledgeItem


@runtime_checkable
class BaseSourceConnector(Protocol):
    """
    Read-only protocol for source data connectors.

    Every method receives explicit context so connectors never
    rely on ambient/global state.
    """

    @property
    def connector_name(self) -> str:
        """Short human-readable identifier, e.g. ``'internal-api'``."""
        ...

    @property
    def source_type(self) -> str:
        """Connector category, e.g. ``'internal_api'``, ``'database'``."""
        ...

    async def test_connection(self) -> bool:
        """
        Verify that the upstream source is reachable.

        Returns True on success, False on failure.
        Implementations should catch and log their own exceptions.
        """
        ...

    async def fetch_item_refs(
        self,
        *,
        tenant_id: str,
        source_key: str,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Return lightweight references for items available in the source.

        Each reference is a dict with at least ``{"external_id": "..."}``
        and any additional hints the connector wants to pass through
        (e.g. ``updated_at``, ``title``).

        Args:
            tenant_id:  Tenant scope for the fetch.
            source_key: Identifier of the configured source definition.
            params:     Optional connector-specific query parameters.
        """
        ...

    async def fetch_item_detail(
        self,
        *,
        tenant_id: str,
        source_key: str,
        external_id: str,
        ref: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """
        Fetch the full payload for a single item.

        Returns ``None`` if the item is not found or inaccessible.

        Args:
            tenant_id:   Tenant scope.
            source_key:  Source definition identifier.
            external_id: The item's unique key in the source system.
            ref:         The original reference dict from ``fetch_item_refs``
                         (may contain extra context the connector can reuse).
        """
        ...

    def map_to_canonical_item(
        self,
        *,
        source_key: str,
        raw_detail: dict[str, Any],
    ) -> CanonicalKnowledgeItem | None:
        """
        Transform a raw detail payload into a ``CanonicalKnowledgeItem``.

        Returns ``None`` if the payload cannot be meaningfully mapped
        (e.g. missing required fields, unsupported format).

        This is a pure mapping function — no I/O, no side-effects.
        """
        ...
