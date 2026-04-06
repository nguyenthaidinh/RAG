from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseIngestStrategy(ABC):
    @abstractmethod
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
        raise NotImplementedError