"""
Mock source connector for development and testing.

Generates synthetic items so the sync flow can be exercised end-to-end
without a real upstream system.  Not intended for production use.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from app.services.source_platform.canonical_item import CanonicalKnowledgeItem


class MockSourceConnector:
    """
    Implements ``BaseSourceConnector`` protocol with in-memory mock data.
    """

    @property
    def connector_name(self) -> str:
        return "mock"

    @property
    def source_type(self) -> str:
        return "mock"

    async def test_connection(self) -> bool:
        return True

    async def fetch_item_refs(
        self,
        *,
        tenant_id: str,
        source_key: str,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return a small set of synthetic item refs."""
        count = (params or {}).get("count", 3)
        return [
            {
                "external_id": f"mock-{i}",
                "title": f"Mock Item {i}",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            for i in range(1, int(count) + 1)
        ]

    async def fetch_item_detail(
        self,
        *,
        tenant_id: str,
        source_key: str,
        external_id: str,
        ref: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Return a synthetic detail payload."""
        return {
            "external_id": external_id,
            "title": (ref or {}).get("title", f"Mock {external_id}"),
            "body_text": f"This is mock content for item {external_id}. "
            f"Generated at {datetime.now(timezone.utc).isoformat()}.",
            "summary": f"Summary of {external_id}",
            "source_uri": f"mock://items/{external_id}",
            "updated_at": (ref or {}).get("updated_at"),
            "metadata": {"mock_run_id": str(uuid.uuid4())[:8]},
        }

    def map_to_canonical_item(
        self,
        *,
        source_key: str,
        raw_detail: dict[str, Any],
    ) -> CanonicalKnowledgeItem | None:
        """Map mock detail to a CanonicalKnowledgeItem."""
        external_id = raw_detail.get("external_id")
        body_text = raw_detail.get("body_text")
        if not external_id or not body_text:
            return None

        updated_str = raw_detail.get("updated_at")
        updated_at = None
        if updated_str:
            try:
                updated_at = datetime.fromisoformat(updated_str)
            except (ValueError, TypeError):
                pass

        return CanonicalKnowledgeItem(
            external_id=external_id,
            source_key=source_key,
            source_type=self.source_type,
            title=raw_detail.get("title"),
            body_text=body_text,
            summary=raw_detail.get("summary"),
            source_uri=raw_detail.get("source_uri"),
            updated_at=updated_at,
            metadata=raw_detail.get("metadata", {}),
        )
