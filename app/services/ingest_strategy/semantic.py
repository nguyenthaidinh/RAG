from __future__ import annotations

from typing import Any

from app.services.document_metadata_builder import DocumentMetadataBuilder
from app.services.ingest_strategy.base import BaseIngestStrategy


class SemanticIngestStrategy(BaseIngestStrategy):
    def __init__(self) -> None:
        self._builder = DocumentMetadataBuilder()

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
        semantic_metadata = self._builder.build(
            title=title,
            text=text,
            file_name=file_name,
            original_name=original_name,
            content_type=content_type,
            size_bytes=size_bytes,
            ingest_via=ingest_via,
        )

        semantic_metadata["user_metadata"] = raw_metadata or {}
        return semantic_metadata