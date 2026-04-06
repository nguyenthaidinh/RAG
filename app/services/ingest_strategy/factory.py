from __future__ import annotations

from app.schemas.ingest_mode import IngestMode
from app.services.ingest_strategy.base import BaseIngestStrategy
from app.services.ingest_strategy.legacy import LegacyIngestStrategy
from app.services.ingest_strategy.semantic import SemanticIngestStrategy


class IngestStrategyFactory:
    @staticmethod
    def create(mode: IngestMode) -> BaseIngestStrategy:
        if mode == IngestMode.SEMANTIC:
            return SemanticIngestStrategy()
        return LegacyIngestStrategy()