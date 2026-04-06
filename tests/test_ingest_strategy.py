from app.schemas.ingest_mode import IngestMode
from app.services.ingest_strategy.factory import IngestStrategyFactory


def test_factory_returns_legacy():
    strategy = IngestStrategyFactory.create(IngestMode.LEGACY)
    assert strategy.__class__.__name__ == "LegacyIngestStrategy"


def test_factory_returns_semantic():
    strategy = IngestStrategyFactory.create(IngestMode.SEMANTIC)
    assert strategy.__class__.__name__ == "SemanticIngestStrategy"