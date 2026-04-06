"""
Phase 3D — Representation Policy tests.

Tests cover:
  - RepresentationIntent enum
  - RepresentationPreference schema (telemetry, neutral)
  - RepresentationIntentService classification (overview, exact, citation, mixed)
  - Tie-breaking rules
  - Confidence threshold gate
  - Feature flag behavior
  - Fail-open safety
  - Selector boost integration (overview→synthesized, exact→original, citation→original)
  - Selector backward compat (no preference)
  - Config validation
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def _mock_settings(**overrides):
    defaults = {
        "REPRESENTATION_POLICY_ENABLED": True,
        "REPRESENTATION_POLICY_OVERVIEW_SYNTH_WEIGHT": 0.12,
        "REPRESENTATION_POLICY_EXACT_ORIGINAL_WEIGHT": 0.14,
        "REPRESENTATION_POLICY_CITATION_ORIGINAL_WEIGHT": 0.18,
        "REPRESENTATION_POLICY_MIXED_SYNTH_WEIGHT": 0.05,
        "REPRESENTATION_POLICY_CONFIDENCE_THRESHOLD": 0.6,
    }
    defaults.update(overrides)
    mock = MagicMock()
    for k, v in defaults.items():
        setattr(mock, k, v)
    return mock


def _make_svc(**settings_overrides):
    mock = _mock_settings(**settings_overrides)
    with patch("app.services.retrieval.representation_intent_service.settings", mock):
        from app.services.retrieval.representation_intent_service import RepresentationIntentService
        return RepresentationIntentService()


def _make_candidate(doc_id, score, repr_type="original", parent_id=None):
    from app.services.retrieval.document_representation_selector import RetrievalCandidate
    from app.services.retrieval.types import make_chunk_id
    return RetrievalCandidate(
        document_id=doc_id,
        chunk_id=make_chunk_id(doc_id, 0),
        chunk_index=0,
        score=score,
        snippet=f"snippet {doc_id}",
        title=f"Title {doc_id}",
        version_id="v1",
        source="rerank",
        representation_type=repr_type,
        parent_document_id=parent_id,
    )


# ── Schema ────────────────────────────────────────────────────────────


class TestRepresentationSchema:

    def test_all_intents(self):
        from app.schemas.retrieval_representation import RepresentationIntent
        assert RepresentationIntent.OVERVIEW_SUMMARY == "overview_summary"
        assert RepresentationIntent.EXACT_SPECIFIC == "exact_specific"
        assert RepresentationIntent.CITATION_SENSITIVE == "citation_sensitive"
        assert RepresentationIntent.EXPLANATORY_MIXED == "explanatory_mixed"
        assert RepresentationIntent.BALANCED_DEFAULT == "balanced_default"

    def test_neutral(self):
        from app.schemas.retrieval_representation import RepresentationPreference
        pref = RepresentationPreference.neutral()
        assert not pref.has_preference
        assert pref.preferred_type == "balanced"

    def test_has_preference(self):
        from app.schemas.retrieval_representation import RepresentationPreference
        pref = RepresentationPreference(preferred_type="original", strength=0.14)
        assert pref.has_preference

    def test_no_preference_balanced(self):
        from app.schemas.retrieval_representation import RepresentationPreference
        pref = RepresentationPreference(preferred_type="balanced", strength=0.5)
        assert not pref.has_preference

    def test_telemetry_safe(self):
        from app.schemas.retrieval_representation import (
            RepresentationIntent, RepresentationPreference,
        )
        pref = RepresentationPreference(
            intent=RepresentationIntent.CITATION_SENSITIVE,
            preferred_type="original",
            strength=0.18,
            confidence=0.8,
            reason="citation_keywords=3",
        )
        tele = pref.telemetry_dict()
        assert tele["intent"] == "citation_sensitive"
        assert tele["preferred_type"] == "original"
        assert "citation_keywords" not in str(tele)  # reason not in telemetry
        assert tele["has_preference"] is True


# ── Classification ────────────────────────────────────────────────────


class TestClassification:

    def test_overview_vi(self):
        from app.schemas.retrieval_representation import RepresentationIntent
        svc = _make_svc()
        pref = svc.classify("tóm tắt nội dung tài liệu này")
        assert pref.intent == RepresentationIntent.OVERVIEW_SUMMARY
        assert pref.preferred_type == "synthesized"

    def test_overview_en(self):
        from app.schemas.retrieval_representation import RepresentationIntent
        svc = _make_svc()
        pref = svc.classify("give me a summary of the document")
        assert pref.intent == RepresentationIntent.OVERVIEW_SUMMARY
        assert pref.preferred_type == "synthesized"

    def test_exact_vi(self):
        from app.schemas.retrieval_representation import RepresentationIntent
        svc = _make_svc()
        pref = svc.classify("điều khoản nào quy định chi tiết về nghỉ phép")
        assert pref.intent == RepresentationIntent.EXACT_SPECIFIC
        assert pref.preferred_type == "original"

    def test_exact_en(self):
        from app.schemas.retrieval_representation import RepresentationIntent
        svc = _make_svc()
        pref = svc.classify("show me the specific clause about termination")
        assert pref.intent == RepresentationIntent.EXACT_SPECIFIC
        assert pref.preferred_type == "original"

    def test_citation_vi(self):
        from app.schemas.retrieval_representation import RepresentationIntent
        svc = _make_svc()
        pref = svc.classify("trích dẫn nguyên văn quy định nào nói về kỷ luật")
        assert pref.intent == RepresentationIntent.CITATION_SENSITIVE
        assert pref.preferred_type == "original"

    def test_citation_en(self):
        from app.schemas.retrieval_representation import RepresentationIntent
        svc = _make_svc()
        pref = svc.classify("citation verbatim reference from the original document")
        assert pref.intent == RepresentationIntent.CITATION_SENSITIVE
        assert pref.preferred_type == "original"

    def test_citation_has_highest_weight(self):
        svc = _make_svc()
        pref = svc.classify("trích dẫn nguyên văn quy định")
        assert pref.strength == 0.18  # citation weight

    def test_explanatory_vi(self):
        from app.schemas.retrieval_representation import RepresentationIntent
        svc = _make_svc()
        pref = svc.classify("giải thích ý nghĩa của quy định này")
        assert pref.intent == RepresentationIntent.EXPLANATORY_MIXED
        assert pref.preferred_type == "synthesized"

    def test_explanatory_en(self):
        from app.schemas.retrieval_representation import RepresentationIntent
        svc = _make_svc()
        pref = svc.classify("explain the meaning of this policy")
        assert pref.intent == RepresentationIntent.EXPLANATORY_MIXED
        assert pref.preferred_type == "synthesized"

    def test_balanced_default(self):
        svc = _make_svc()
        pref = svc.classify("chính sách nghỉ phép của công ty")
        assert not pref.has_preference

    def test_empty_query(self):
        svc = _make_svc()
        pref = svc.classify("")
        assert not pref.has_preference

    def test_tie_breaking_citation_wins(self):
        """When citation and exact tied, citation (safer) wins."""
        from app.schemas.retrieval_representation import RepresentationIntent
        svc = _make_svc()
        # "nguyên văn" → citation, "chi tiết" → exact
        pref = svc.classify("nguyên văn chi tiết")
        assert pref.preferred_type == "original"

    def test_confidence_threshold(self):
        """Low confidence → neutral."""
        svc = _make_svc(REPRESENTATION_POLICY_CONFIDENCE_THRESHOLD=0.99)
        pref = svc.classify("tóm tắt nội dung")
        # Single keyword match → confidence too low for threshold 0.99
        assert not pref.has_preference


# ── Feature Flag ──────────────────────────────────────────────────────


class TestFeatureFlag:

    def test_disabled_returns_neutral(self):
        svc = _make_svc(REPRESENTATION_POLICY_ENABLED=False)
        pref = svc.classify("tóm tắt nguyên văn")
        assert not pref.has_preference

    def test_enabled_property(self):
        svc = _make_svc(REPRESENTATION_POLICY_ENABLED=True)
        assert svc.enabled is True


# ── Fail Open ─────────────────────────────────────────────────────────


class TestFailOpen:

    def test_classify_error_returns_neutral(self):
        svc = _make_svc()
        # Replace module-level keywords with non-iterable to force error
        import app.services.retrieval.representation_intent_service as mod
        original_kw = mod._OVERVIEW_KW
        mod._OVERVIEW_KW = None  # will crash sum(1 for kw in None ...)
        try:
            pref = svc.classify("tóm tắt nội dung")
            assert not pref.has_preference  # fail-open → neutral
        finally:
            mod._OVERVIEW_KW = original_kw


# ── Selector Integration ──────────────────────────────────────────────


class TestSelectorIntegration:
    """Test that RepresentationPreference affects family selection."""

    def test_no_preference_backward_compat(self):
        """Without preference, selector works as before."""
        from app.services.retrieval.document_representation_selector import (
            DocumentRepresentationSelector,
        )
        selector = DocumentRepresentationSelector(mode="balanced")
        orig = _make_candidate(1, 0.80, "original")
        synth = _make_candidate(2, 0.75, "synthesized", parent_id=1)

        selected = selector.consolidate("test query", [orig, synth])
        # No preference → existing tie-breaking
        assert len(selected) == 1  # single family

    def test_overview_prefers_synthesized(self):
        """Overview preference boosts synthesized to win family selection."""
        from app.schemas.retrieval_representation import (
            RepresentationIntent, RepresentationPreference,
        )
        from app.services.retrieval.document_representation_selector import (
            DocumentRepresentationSelector,
        )
        selector = DocumentRepresentationSelector(mode="balanced")
        # Original slightly higher score
        orig = _make_candidate(1, 0.82, "original")
        synth = _make_candidate(2, 0.78, "synthesized", parent_id=1)

        pref = RepresentationPreference(
            intent=RepresentationIntent.OVERVIEW_SUMMARY,
            preferred_type="synthesized",
            strength=0.12,
            confidence=0.8,
        )

        selected = selector.consolidate(
            "tóm tắt nội dung", [orig, synth],
            representation_preference=pref,
        )
        assert len(selected) == 1
        # Synthesized (0.78 + 0.12 = 0.90) vs original (0.82) → synth wins
        assert selected[0].selected_representation_type == "synthesized"

    def test_exact_prefers_original(self):
        """Exact preference boosts original to hold its lead."""
        from app.schemas.retrieval_representation import (
            RepresentationIntent, RepresentationPreference,
        )
        from app.services.retrieval.document_representation_selector import (
            DocumentRepresentationSelector,
        )
        selector = DocumentRepresentationSelector(mode="balanced")
        orig = _make_candidate(1, 0.78, "original")
        synth = _make_candidate(2, 0.80, "synthesized", parent_id=1)

        pref = RepresentationPreference(
            intent=RepresentationIntent.EXACT_SPECIFIC,
            preferred_type="original",
            strength=0.14,
            confidence=0.8,
        )

        selected = selector.consolidate(
            "điều khoản cụ thể nào", [orig, synth],
            representation_preference=pref,
        )
        assert len(selected) == 1
        # Original (0.78 + 0.14 = 0.92) vs synth (0.80) → original wins
        assert selected[0].selected_representation_type == "original"

    def test_citation_strongly_prefers_original(self):
        """Citation preference with highest weight keeps original."""
        from app.schemas.retrieval_representation import (
            RepresentationIntent, RepresentationPreference,
        )
        from app.services.retrieval.document_representation_selector import (
            DocumentRepresentationSelector,
        )
        selector = DocumentRepresentationSelector(mode="balanced")
        orig = _make_candidate(1, 0.70, "original")
        synth = _make_candidate(2, 0.82, "synthesized", parent_id=1)

        pref = RepresentationPreference(
            intent=RepresentationIntent.CITATION_SENSITIVE,
            preferred_type="original",
            strength=0.18,
            confidence=0.9,
        )

        selected = selector.consolidate(
            "trích dẫn nguyên văn", [orig, synth],
            representation_preference=pref,
        )
        assert len(selected) == 1
        # Original (0.70 + 0.18 = 0.88) vs synth (0.82) → original wins
        assert selected[0].selected_representation_type == "original"

    def test_balanced_no_boost(self):
        """Balanced preference doesn't boost either type."""
        from app.schemas.retrieval_representation import RepresentationPreference
        from app.services.retrieval.document_representation_selector import (
            DocumentRepresentationSelector,
        )
        selector = DocumentRepresentationSelector(mode="balanced")
        orig = _make_candidate(1, 0.80, "original")
        synth = _make_candidate(2, 0.75, "synthesized", parent_id=1)

        pref = RepresentationPreference.neutral()

        selected = selector.consolidate(
            "test query", [orig, synth],
            representation_preference=pref,
        )
        assert len(selected) == 1
        # No boost → normal behavior

    def test_single_member_family_unaffected(self):
        """Single-member families are selected unconditionally."""
        from app.schemas.retrieval_representation import (
            RepresentationIntent, RepresentationPreference,
        )
        from app.services.retrieval.document_representation_selector import (
            DocumentRepresentationSelector,
        )
        selector = DocumentRepresentationSelector(mode="balanced")
        orig = _make_candidate(1, 0.80, "original")  # no synthesized child

        pref = RepresentationPreference(
            intent=RepresentationIntent.OVERVIEW_SUMMARY,
            preferred_type="synthesized",
            strength=0.12,
            confidence=0.8,
        )

        selected = selector.consolidate(
            "tóm tắt", [orig],
            representation_preference=pref,
        )
        assert len(selected) == 1
        assert selected[0].document_id == 1  # only option


# ── Config ────────────────────────────────────────────────────────────


class TestConfig:

    def test_all_keys_exist(self):
        from app.core.config import Settings
        expected = [
            "REPRESENTATION_POLICY_ENABLED",
            "REPRESENTATION_POLICY_OVERVIEW_SYNTH_WEIGHT",
            "REPRESENTATION_POLICY_EXACT_ORIGINAL_WEIGHT",
            "REPRESENTATION_POLICY_CITATION_ORIGINAL_WEIGHT",
            "REPRESENTATION_POLICY_MIXED_SYNTH_WEIGHT",
            "REPRESENTATION_POLICY_CONFIDENCE_THRESHOLD",
        ]
        for key in expected:
            assert key in Settings.model_fields, f"Missing config: {key}"

    def test_defaults_conservative(self):
        from app.core.config import Settings
        fields = Settings.model_fields
        assert fields["REPRESENTATION_POLICY_ENABLED"].default is False
        assert fields["REPRESENTATION_POLICY_CITATION_ORIGINAL_WEIGHT"].default <= 0.20
