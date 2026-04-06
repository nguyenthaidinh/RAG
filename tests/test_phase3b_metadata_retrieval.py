"""
Phase 3B — Metadata-Aware Retrieval tests.

Tests cover:
  - MetadataPreference schema (whitelist, telemetry, empty)
  - MetadataIntentService parsing (repr type, source, title, recency)
  - MetadataBiasReranker scoring (source, repr, title, recency bias)
  - Feature flag gate
  - Fail-open on parse error
  - Cross-dedupe / safety
  - Config validation
  - End-to-end bias application
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import pytest


def _mock_settings(**overrides):
    defaults = {
        "METADATA_RETRIEVAL_ENABLED": True,
        "METADATA_RETRIEVAL_CONFIDENCE_THRESHOLD": 0.6,
        "METADATA_RETRIEVAL_MAX_TITLE_TERMS": 3,
        "METADATA_RETRIEVAL_MAX_TAGS": 3,
        "METADATA_RETRIEVAL_SOURCE_BIAS_WEIGHT": 0.08,
        "METADATA_RETRIEVAL_REPRESENTATION_BIAS_WEIGHT": 0.10,
        "METADATA_RETRIEVAL_TITLE_BIAS_WEIGHT": 0.06,
        "METADATA_RETRIEVAL_RECENCY_BIAS_WEIGHT": 0.05,
    }
    defaults.update(overrides)
    mock = MagicMock()
    for k, v in defaults.items():
        setattr(mock, k, v)
    return mock


def _make_intent_svc(**settings_overrides):
    mock = _mock_settings(**settings_overrides)
    with patch("app.services.retrieval.metadata_intent_service.settings", mock):
        from app.services.retrieval.metadata_intent_service import MetadataIntentService
        return MetadataIntentService()


def _make_bias_reranker(**settings_overrides):
    mock = _mock_settings(**settings_overrides)
    with patch("app.services.retrieval.metadata_bias.settings", mock):
        from app.services.retrieval.metadata_bias import MetadataBiasReranker
        return MetadataBiasReranker()


def _make_chunk(doc_id, score, title=None, chunk_index=0):
    from app.services.retrieval.types import ScoredChunk, make_chunk_id
    return ScoredChunk(
        chunk_id=make_chunk_id(doc_id, chunk_index),
        document_id=doc_id,
        version_id="v1",
        chunk_index=chunk_index,
        score=score,
        source="rerank",
        snippet=f"snippet for doc {doc_id}",
        title=title,
    )


# ── MetadataPreference Schema ─────────────────────────────────────────


class TestMetadataPreference:

    def test_empty(self):
        from app.schemas.retrieval_metadata import MetadataPreference
        pref = MetadataPreference.empty()
        assert not pref.has_preferences
        assert pref.confidence == 0.0

    def test_has_preferences_with_sources(self):
        from app.schemas.retrieval_metadata import MetadataPreference
        pref = MetadataPreference(preferred_sources=("form",))
        assert pref.has_preferences

    def test_has_preferences_with_repr(self):
        from app.schemas.retrieval_metadata import MetadataPreference
        pref = MetadataPreference(preferred_representation_types=("synthesized",))
        assert pref.has_preferences

    def test_has_preferences_with_title(self):
        from app.schemas.retrieval_metadata import MetadataPreference
        pref = MetadataPreference(preferred_title_terms=("policy",))
        assert pref.has_preferences

    def test_has_preferences_with_recency(self):
        from app.schemas.retrieval_metadata import MetadataPreference
        pref = MetadataPreference(prefer_newest=True)
        assert pref.has_preferences

    def test_telemetry_safe(self):
        from app.schemas.retrieval_metadata import MetadataPreference
        pref = MetadataPreference(
            preferred_sources=("confidential_source",),
            preferred_title_terms=("secret_doc",),
            confidence=0.8,
        )
        tele = pref.telemetry_dict()
        assert "confidential" not in str(tele)
        assert "secret" not in str(tele)
        assert tele["source_count"] == 1
        assert tele["title_term_count"] == 1
        assert tele["confidence"] == 0.8

    def test_telemetry_keys(self):
        from app.schemas.retrieval_metadata import MetadataPreference
        tele = MetadataPreference.empty().telemetry_dict()
        expected = {
            "has_preferences", "source_count", "repr_type_count",
            "title_term_count", "prefer_newest", "confidence",
        }
        assert set(tele.keys()) == expected

    def test_whitelist_fields(self):
        from app.schemas.retrieval_metadata import SUPPORTED_FIELDS
        assert "source" in SUPPORTED_FIELDS
        assert "title" in SUPPORTED_FIELDS
        assert "representation_type" in SUPPORTED_FIELDS
        # unsupported fields NOT in whitelist
        assert "password" not in SUPPORTED_FIELDS
        assert "tenant_id" not in SUPPORTED_FIELDS


# ── MetadataIntentService ──────────────────────────────────────────────


class TestMetadataIntentService:

    def test_disabled_returns_empty(self):
        svc = _make_intent_svc(METADATA_RETRIEVAL_ENABLED=False)
        pref = svc.parse("mẫu đơn đăng ký")
        assert not pref.has_preferences

    def test_empty_query(self):
        svc = _make_intent_svc()
        pref = svc.parse("")
        assert not pref.has_preferences

    def test_detect_summary_repr(self):
        svc = _make_intent_svc()
        pref = svc.parse("bản tóm tắt quy định tuyển sinh")
        assert "synthesized" in pref.preferred_representation_types

    def test_detect_original_repr(self):
        svc = _make_intent_svc()
        pref = svc.parse("cho tôi xem nguyên văn tài liệu")
        assert "original" in pref.preferred_representation_types

    def test_ambiguous_repr_returns_empty(self):
        svc = _make_intent_svc()
        pref = svc.parse("tóm tắt bản gốc chi tiết")
        # both summary and original keywords → ambiguous → no pref
        assert len(pref.preferred_representation_types) == 0

    def test_detect_form_source(self):
        svc = _make_intent_svc()
        pref = svc.parse("mẫu đơn đăng ký nhập học")
        assert "form" in pref.preferred_sources

    def test_detect_regulation_source(self):
        svc = _make_intent_svc()
        pref = svc.parse("quy chế đào tạo hiện hành")
        assert "regulation" in pref.preferred_sources

    def test_detect_guide_source(self):
        svc = _make_intent_svc()
        pref = svc.parse("hướng dẫn nộp hồ sơ")
        assert "guide" in pref.preferred_sources

    def test_detect_notice_source(self):
        svc = _make_intent_svc()
        pref = svc.parse("thông báo tuyển sinh 2025")
        assert "notice" in pref.preferred_sources

    def test_detect_recency(self):
        svc = _make_intent_svc()
        pref = svc.parse("quy định mới nhất về tuyển sinh")
        assert pref.prefer_newest is True

    def test_detect_recency_en(self):
        svc = _make_intent_svc()
        pref = svc.parse("show me the latest policy")
        assert pref.prefer_newest is True

    def test_title_term_quoted(self):
        svc = _make_intent_svc()
        pref = svc.parse('tìm tài liệu "quy chế 2025" cho tôi')
        assert "quy chế 2025" in pref.preferred_title_terms

    def test_multiple_signals_higher_confidence(self):
        svc = _make_intent_svc()
        pref1 = svc.parse("mẫu đơn")  # 1 signal (source)
        pref2 = svc.parse("mẫu đơn mới nhất")  # 2 signals (source + recency)
        assert pref2.confidence > pref1.confidence

    def test_no_signals_returns_empty(self):
        svc = _make_intent_svc()
        pref = svc.parse("xin chào bạn")
        assert not pref.has_preferences

    def test_fail_open_on_error(self):
        svc = _make_intent_svc()
        # Patch internal parse to raise
        with patch.object(svc, "_detect_representation_type", side_effect=Exception("boom")):
            pref = svc.parse("mẫu đơn đăng ký")
        assert not pref.has_preferences

    def test_enabled_property(self):
        svc = _make_intent_svc(METADATA_RETRIEVAL_ENABLED=True)
        assert svc.enabled is True


# ── MetadataBiasReranker ───────────────────────────────────────────────


class TestMetadataBiasReranker:

    def test_no_preference_returns_unchanged(self):
        from app.schemas.retrieval_metadata import MetadataPreference
        reranker = _make_bias_reranker()
        chunks = [_make_chunk(1, 0.8), _make_chunk(2, 0.7)]
        result = reranker.apply_bias(
            chunks=chunks,
            preference=MetadataPreference.empty(),
            doc_metadata={},
        )
        assert [c.score for c in result] == [0.8, 0.7]

    def test_source_bias_applied(self):
        from app.schemas.retrieval_metadata import MetadataPreference
        reranker = _make_bias_reranker()
        chunks = [_make_chunk(1, 0.8), _make_chunk(2, 0.75)]
        pref = MetadataPreference(preferred_sources=("form",), confidence=0.8)
        doc_meta = {
            1: {"source": "upload", "title": "Doc A", "representation_type": "original", "meta": {}, "created_at": None},
            2: {"source": "form-service", "title": "Mẫu đơn B", "representation_type": "original", "meta": {}, "created_at": None},
        }
        result = reranker.apply_bias(chunks=chunks, preference=pref, doc_metadata=doc_meta)
        # Doc 2 should get source bias boost
        doc2_score = next(c.score for c in result if c.document_id == 2)
        assert doc2_score > 0.75  # boosted

    def test_repr_type_bias(self):
        from app.schemas.retrieval_metadata import MetadataPreference
        reranker = _make_bias_reranker()
        chunks = [_make_chunk(1, 0.8), _make_chunk(2, 0.75)]
        pref = MetadataPreference(preferred_representation_types=("synthesized",), confidence=0.8)
        doc_meta = {
            1: {"source": "x", "title": "A", "representation_type": "original", "meta": {}, "created_at": None},
            2: {"source": "x", "title": "B", "representation_type": "synthesized", "meta": {}, "created_at": None},
        }
        result = reranker.apply_bias(chunks=chunks, preference=pref, doc_metadata=doc_meta)
        doc2_score = next(c.score for c in result if c.document_id == 2)
        assert doc2_score > 0.75  # repr bias boost

    def test_title_term_bias(self):
        from app.schemas.retrieval_metadata import MetadataPreference
        reranker = _make_bias_reranker()
        chunks = [_make_chunk(1, 0.8), _make_chunk(2, 0.75)]
        pref = MetadataPreference(preferred_title_terms=("tuyển sinh",), confidence=0.8)
        doc_meta = {
            1: {"source": "x", "title": "Quy chế đào tạo", "representation_type": "original", "meta": {}, "created_at": None},
            2: {"source": "x", "title": "Quy định tuyển sinh 2025", "representation_type": "original", "meta": {}, "created_at": None},
        }
        result = reranker.apply_bias(chunks=chunks, preference=pref, doc_metadata=doc_meta)
        doc2_score = next(c.score for c in result if c.document_id == 2)
        assert doc2_score > 0.75  # title bias boost

    def test_recency_bias_new_doc(self):
        from app.schemas.retrieval_metadata import MetadataPreference
        reranker = _make_bias_reranker()
        chunks = [_make_chunk(1, 0.8), _make_chunk(2, 0.75)]
        now = datetime.now(timezone.utc)
        pref = MetadataPreference(prefer_newest=True, confidence=0.8)
        doc_meta = {
            1: {"source": "x", "title": "A", "representation_type": "original", "meta": {}, "created_at": now - timedelta(days=365)},
            2: {"source": "x", "title": "B", "representation_type": "original", "meta": {}, "created_at": now - timedelta(days=5)},
        }
        result = reranker.apply_bias(chunks=chunks, preference=pref, doc_metadata=doc_meta)
        doc2_score = next(c.score for c in result if c.document_id == 2)
        assert doc2_score > 0.75  # recency boost

    def test_recency_bias_medium_age(self):
        from app.schemas.retrieval_metadata import MetadataPreference
        reranker = _make_bias_reranker()
        chunks = [_make_chunk(1, 0.8)]
        now = datetime.now(timezone.utc)
        pref = MetadataPreference(prefer_newest=True, confidence=0.8)
        doc_meta = {
            1: {"source": "x", "title": "A", "representation_type": "original", "meta": {}, "created_at": now - timedelta(days=60)},
        }
        result = reranker.apply_bias(chunks=chunks, preference=pref, doc_metadata=doc_meta)
        # 60 days → half boost
        assert result[0].score > 0.8
        assert result[0].score < 0.8 + 0.05  # less than full boost

    def test_multiple_biases_stack(self):
        from app.schemas.retrieval_metadata import MetadataPreference
        reranker = _make_bias_reranker()
        chunks = [_make_chunk(1, 0.7)]
        now = datetime.now(timezone.utc)
        pref = MetadataPreference(
            preferred_sources=("form",),
            preferred_representation_types=("synthesized",),
            prefer_newest=True,
            confidence=0.9,
        )
        doc_meta = {
            1: {"source": "form-svc", "title": "Form A", "representation_type": "synthesized", "meta": {}, "created_at": now - timedelta(days=10)},
        }
        result = reranker.apply_bias(chunks=chunks, preference=pref, doc_metadata=doc_meta)
        total_bias = result[0].score - 0.7
        # source(0.08) + repr(0.10) + recency(0.05) = 0.23
        assert total_bias > 0.20  # multiple biases stacked

    def test_bias_reorders_results(self):
        from app.schemas.retrieval_metadata import MetadataPreference
        reranker = _make_bias_reranker()
        chunks = [_make_chunk(1, 0.80), _make_chunk(2, 0.75)]
        now = datetime.now(timezone.utc)
        pref = MetadataPreference(
            preferred_sources=("special",),
            preferred_representation_types=("synthesized",),
            prefer_newest=True,
            confidence=0.9,
        )
        doc_meta = {
            1: {"source": "generic", "title": "A", "representation_type": "original", "meta": {}, "created_at": now - timedelta(days=365)},
            2: {"source": "special", "title": "B", "representation_type": "synthesized", "meta": {}, "created_at": now - timedelta(days=5)},
        }
        result = reranker.apply_bias(chunks=chunks, preference=pref, doc_metadata=doc_meta)
        # Doc 2 was lower but has all biases → should overtake doc 1
        assert result[0].document_id == 2

    def test_empty_chunks(self):
        from app.schemas.retrieval_metadata import MetadataPreference
        reranker = _make_bias_reranker()
        result = reranker.apply_bias(
            chunks=[],
            preference=MetadataPreference(preferred_sources=("x",)),
            doc_metadata={},
        )
        assert result == []

    def test_missing_metadata_no_crash(self):
        from app.schemas.retrieval_metadata import MetadataPreference
        reranker = _make_bias_reranker()
        chunks = [_make_chunk(1, 0.8)]
        pref = MetadataPreference(preferred_sources=("form",), confidence=0.8)
        # empty metadata → no crash, no bias
        result = reranker.apply_bias(chunks=chunks, preference=pref, doc_metadata={})
        assert result[0].score == 0.8

    def test_category_match_in_meta_json(self):
        from app.schemas.retrieval_metadata import MetadataPreference
        reranker = _make_bias_reranker()
        chunks = [_make_chunk(1, 0.8)]
        pref = MetadataPreference(preferred_sources=("form",), confidence=0.8)
        doc_meta = {
            1: {"source": "upload", "title": "Doc", "representation_type": "original", "meta": {"category": "form-template"}, "created_at": None},
        }
        result = reranker.apply_bias(chunks=chunks, preference=pref, doc_metadata=doc_meta)
        assert result[0].score > 0.8  # category match in meta json


# ── Feature Flag ───────────────────────────────────────────────────────


class TestFeatureFlag:

    def test_disabled_returns_no_pref(self):
        svc = _make_intent_svc(METADATA_RETRIEVAL_ENABLED=False)
        pref = svc.parse("mẫu đơn mới nhất")
        assert not pref.has_preferences

    def test_flag_off_behavior_same_as_3a(self):
        """When flag off, entire metadata path is skipped."""
        svc = _make_intent_svc(METADATA_RETRIEVAL_ENABLED=False)
        pref = svc.parse("bản tóm tắt quy định latest")
        assert pref.confidence == 0.0
        assert not pref.has_preferences


# ── Config ────────────────────────────────────────────────────────────


class TestConfig:

    def test_all_config_keys_exist(self):
        from app.core.config import Settings
        expected = [
            "METADATA_RETRIEVAL_ENABLED",
            "METADATA_RETRIEVAL_CONFIDENCE_THRESHOLD",
            "METADATA_RETRIEVAL_MAX_TITLE_TERMS",
            "METADATA_RETRIEVAL_MAX_TAGS",
            "METADATA_RETRIEVAL_SOURCE_BIAS_WEIGHT",
            "METADATA_RETRIEVAL_REPRESENTATION_BIAS_WEIGHT",
            "METADATA_RETRIEVAL_TITLE_BIAS_WEIGHT",
            "METADATA_RETRIEVAL_RECENCY_BIAS_WEIGHT",
        ]
        for key in expected:
            assert key in Settings.model_fields, f"Missing config: {key}"

    def test_defaults_conservative(self):
        from app.core.config import Settings
        fields = Settings.model_fields
        assert fields["METADATA_RETRIEVAL_ENABLED"].default is False
        assert fields["METADATA_RETRIEVAL_SOURCE_BIAS_WEIGHT"].default <= 0.10
        assert fields["METADATA_RETRIEVAL_REPRESENTATION_BIAS_WEIGHT"].default <= 0.15
