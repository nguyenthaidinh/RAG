"""
Source Connector Platform — multi-source knowledge ingestion.

Provides a connector-based abstraction layer for ingesting knowledge
from external systems (Internal API, databases, HTML pages, etc.)
into the existing document pipeline.

Phase 1/2: skeleton + InternalApiConnector.
Phase 3:   config-driven onboarding for multiple sources.

Public API::

    from app.services.source_platform import (
        CanonicalKnowledgeItem,
        BaseSourceConnector,
        SourceRegistry,
        SourceSyncService,
        OnboardedSourceConfig,
        sync_onboarded_source,
    )
"""
from app.services.source_platform.base_source_connector import BaseSourceConnector
from app.services.source_platform.canonical_item import CanonicalKnowledgeItem
from app.services.source_platform.onboarded_source_config import (
    OnboardedSourceConfig,
)
from app.services.source_platform.source_onboarding_service import (
    sync_onboarded_source,
)
from app.services.source_platform.source_registry import SourceRegistry
from app.services.source_platform.source_sync_service import SourceSyncService

__all__ = [
    "BaseSourceConnector",
    "CanonicalKnowledgeItem",
    "OnboardedSourceConfig",
    "SourceRegistry",
    "SourceSyncService",
    "sync_onboarded_source",
]
