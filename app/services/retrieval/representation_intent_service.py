"""
Representation Intent Service (Phase 3D).

Classifies query representation need and produces a
RepresentationPreference for family selection.

Design:
  - Feature-flagged via REPRESENTATION_POLICY_ENABLED
  - Pure heuristics — no LLM, no DB
  - Fail-open: error → RepresentationPreference.neutral()
  - Weights from config
"""
from __future__ import annotations

import logging

from app.core.config import settings
from app.schemas.retrieval_representation import (
    RepresentationIntent,
    RepresentationPreference,
)

logger = logging.getLogger(__name__)


# ── Keyword tables ────────────────────────────────────────────────────

_OVERVIEW_KW = (
    "tóm tắt", "tổng quan", "overview", "summary", "summarize",
    "key points", "main idea", "giới thiệu", "nội dung chính",
    "điểm chính", "khái quát", "tổng hợp",
)

_EXACT_KW = (
    "điều khoản", "khoản nào", "mục nào", "chương nào",
    "chi tiết", "cụ thể", "chính xác", "exact", "specific",
    "detailed", "step by step", "các bước",
    "clause", "section", "article", "paragraph",
)

_CITATION_KW = (
    "nguyên văn", "trích dẫn", "nguồn nào", "văn bản nào",
    "quy định nào", "citation", "verbatim", "quote",
    "according to", "reference", "source",
    "bản gốc", "original", "chứng minh",
)

_EXPLANATORY_KW = (
    "giải thích", "explain", "hiểu", "understand", "meaning",
    "ý nghĩa", "tại sao", "why", "how does", "làm sao",
    "nói rõ hơn", "clarify", "elaborate",
)


class RepresentationIntentService:
    """
    Classifies query representation intent.

    Public API:
      - classify(query_text) → RepresentationPreference
    """

    def __init__(self) -> None:
        self._enabled = getattr(settings, "REPRESENTATION_POLICY_ENABLED", False)
        self._overview_weight = float(
            getattr(settings, "REPRESENTATION_POLICY_OVERVIEW_SYNTH_WEIGHT", 0.12)
        )
        self._exact_weight = float(
            getattr(settings, "REPRESENTATION_POLICY_EXACT_ORIGINAL_WEIGHT", 0.14)
        )
        self._citation_weight = float(
            getattr(settings, "REPRESENTATION_POLICY_CITATION_ORIGINAL_WEIGHT", 0.18)
        )
        self._mixed_weight = float(
            getattr(settings, "REPRESENTATION_POLICY_MIXED_SYNTH_WEIGHT", 0.05)
        )
        self._confidence_threshold = float(
            getattr(settings, "REPRESENTATION_POLICY_CONFIDENCE_THRESHOLD", 0.6)
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    def classify(self, query_text: str) -> RepresentationPreference:
        """
        Classify query and produce representation preference.

        Fail-open: any error → RepresentationPreference.neutral().
        """
        if not self._enabled:
            return RepresentationPreference.neutral()

        if not query_text or not query_text.strip():
            return RepresentationPreference.neutral()

        try:
            q = query_text.lower().strip()

            # Count keyword matches per category
            overview_score = sum(1 for kw in _OVERVIEW_KW if kw in q)
            exact_score = sum(1 for kw in _EXACT_KW if kw in q)
            citation_score = sum(1 for kw in _CITATION_KW if kw in q)
            explanatory_score = sum(1 for kw in _EXPLANATORY_KW if kw in q)

            total = overview_score + exact_score + citation_score + explanatory_score

            if total == 0:
                return RepresentationPreference.neutral()

            # Determine dominant intent
            scores = {
                RepresentationIntent.OVERVIEW_SUMMARY: overview_score,
                RepresentationIntent.EXACT_SPECIFIC: exact_score,
                RepresentationIntent.CITATION_SENSITIVE: citation_score,
                RepresentationIntent.EXPLANATORY_MIXED: explanatory_score,
            }
            best_intent = max(scores, key=scores.get)
            best_score = scores[best_intent]

            # Check for ambiguity: if top two are tied, use mixed
            sorted_scores = sorted(scores.values(), reverse=True)
            if len(sorted_scores) >= 2 and sorted_scores[0] == sorted_scores[1]:
                # Tie between categories
                # If citation is one of the tied → prefer original (safer)
                if citation_score == best_score:
                    best_intent = RepresentationIntent.CITATION_SENSITIVE
                elif exact_score == best_score:
                    best_intent = RepresentationIntent.EXACT_SPECIFIC
                else:
                    best_intent = RepresentationIntent.EXPLANATORY_MIXED

            # Convert intent → preference
            confidence = min(0.5 + best_score * 0.15, 0.95)

            if best_intent == RepresentationIntent.OVERVIEW_SUMMARY:
                pref = RepresentationPreference(
                    intent=best_intent,
                    preferred_type="synthesized",
                    strength=self._overview_weight,
                    confidence=confidence,
                    reason=f"overview_keywords={overview_score}",
                )
            elif best_intent == RepresentationIntent.EXACT_SPECIFIC:
                pref = RepresentationPreference(
                    intent=best_intent,
                    preferred_type="original",
                    strength=self._exact_weight,
                    confidence=confidence,
                    reason=f"exact_keywords={exact_score}",
                )
            elif best_intent == RepresentationIntent.CITATION_SENSITIVE:
                pref = RepresentationPreference(
                    intent=best_intent,
                    preferred_type="original",
                    strength=self._citation_weight,
                    confidence=confidence,
                    reason=f"citation_keywords={citation_score}",
                )
            elif best_intent == RepresentationIntent.EXPLANATORY_MIXED:
                pref = RepresentationPreference(
                    intent=best_intent,
                    preferred_type="synthesized",
                    strength=self._mixed_weight,
                    confidence=confidence,
                    reason=f"explanatory_keywords={explanatory_score}",
                )
            else:
                pref = RepresentationPreference.neutral()

            # Low confidence → neutral
            if pref.confidence < self._confidence_threshold:
                return RepresentationPreference.neutral()

            logger.info(
                "representation_intent.classified intent=%s preferred=%s "
                "strength=%.3f confidence=%.2f",
                pref.intent.value, pref.preferred_type,
                pref.strength, pref.confidence,
            )

            return pref

        except Exception:
            logger.warning("representation_intent.classify_error", exc_info=True)
            return RepresentationPreference.neutral()
