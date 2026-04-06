from __future__ import annotations

from typing import Any

from app.services.ingest_strategy.base import BaseIngestStrategy


class LegacyIngestStrategy(BaseIngestStrategy):
    def build_metadata(
        self,
        *,
        title: str | None,
        text: str,
        file_name: str | None,
        original_name: str | None,
        content_type: str | None,
        size_bytes: int | None,
        ingest_via: str,
        raw_metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        user_metadata = raw_metadata or {}

        return {
            "system": {
                "pipeline_mode": "legacy",
                "pipeline_version": "legacy_v1",
                "ingest_via": ingest_via,
                "file_name": file_name,
                "original_name": original_name,
                "content_type": content_type,
                "size_bytes": size_bytes,
            },
            "user_metadata": user_metadata,
        }