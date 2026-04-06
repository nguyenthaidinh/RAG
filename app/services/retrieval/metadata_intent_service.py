"""
Metadata Intent Service (Phase 3B).

Parses metadata-related intent from query text and produces
a MetadataPreference for downstream retrieval bias.

Design rules:
  - Feature-flagged via METADATA_RETRIEVAL_ENABLED
  - Whitelist-only: only supported Document model fields
  - No hallucinated fields
  - No DB queries — pure text heuristics
  - Fail-open: any error → MetadataPreference.empty()
  - No raw content in logs
"""
from __future__ import annotations

import logging
import re
from typing import Any

from app.core.config import settings
from app.schemas.retrieval_metadata import SUPPORTED_FIELDS, MetadataPreference

logger = logging.getLogger(__name__)


# ── Keyword → metadata mapping tables ─────────────────────────────────

# representation_type preferences
_SUMMARY_KW = (
    "tóm tắt", "bản tóm tắt", "summary", "summarize",
    "tổng hợp", "khái quát", "overview",
)
_ORIGINAL_KW = (
    "nguyên văn", "bản gốc", "original", "verbatim",
    "chi tiết", "exact", "full text",
)

# source-like preferences (document category/type)
_FORM_KW = (
    "mẫu đơn", "biểu mẫu", "form", "template",
    "đơn xin", "mẫu", "phiếu",
)
_REGULATION_KW = (
    "quy chế", "quy định", "nội quy", "quy tắc",
    "regulation", "policy", "rule",
)
_GUIDE_KW = (
    "hướng dẫn", "chỉ dẫn", "guide", "guideline",
    "instruction", "manual", "sổ tay",
)
_NOTICE_KW = (
    "thông báo", "thông tin", "notice", "announcement",
    "công văn", "văn bản",
)

# recency preferences
_RECENCY_KW = (
    "mới nhất", "gần đây", "latest", "newest",
    "most recent", "phiên bản mới", "cập nhật",
    "hiện hành", "current",
)

# Map keyword groups → source label for bias
_SOURCE_KEYWORD_MAP: list[tuple[tuple[str, ...], str]] = [
    (_FORM_KW, "form"),
    (_REGULATION_KW, "regulation"),
    (_GUIDE_KW, "guide"),
    (_NOTICE_KW, "notice"),
]


class MetadataIntentService:
    """
    Parses metadata intent from query text.

    Public API:
      - parse(query_text, rewrite_plan=None) → MetadataPreference
    """

    def __init__(self) -> None:
        self._enabled = getattr(settings, "METADATA_RETRIEVAL_ENABLED", False)
        self._confidence_threshold = float(
            getattr(settings, "METADATA_RETRIEVAL_CONFIDENCE_THRESHOLD", 0.6)
        )
        self._max_title_terms = int(
            getattr(settings, "METADATA_RETRIEVAL_MAX_TITLE_TERMS", 3)
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    def parse(
        self,
        query_text: str,
        *,
        rewrite_plan: Any | None = None,
    ) -> MetadataPreference:
        """
        Parse metadata intent from query.

        Fail-open: any error → MetadataPreference.empty().
        """
        if not self._enabled:
            return MetadataPreference.empty()

        if not query_text or not query_text.strip():
            return MetadataPreference.empty()

        try:
            q = query_text.lower().strip()

            # 1) Representation type preference
            repr_types = self._detect_representation_type(q)

            # 2) Source/category preference
            sources = self._detect_source_category(q)

            # 3) Title term extraction
            title_terms = self._extract_title_terms(q)

            # 4) Recency preference
            prefer_newest = self._detect_recency(q)

            # 5) Calculate confidence
            signals = sum([
                len(repr_types) > 0,
                len(sources) > 0,
                len(title_terms) > 0,
                prefer_newest,
            ])

            if signals == 0:
                return MetadataPreference.empty()

            # More signals → higher confidence
            confidence = min(0.5 + signals * 0.15, 0.95)

            # Build reason
            parts = []
            if repr_types:
                parts.append(f"repr={','.join(repr_types)}")
            if sources:
                parts.append(f"source={','.join(sources)}")
            if title_terms:
                parts.append(f"title_terms={len(title_terms)}")
            if prefer_newest:
                parts.append("recency=true")
            reason = " ".join(parts)

            pref = MetadataPreference(
                preferred_sources=tuple(sources),
                preferred_representation_types=tuple(repr_types),
                preferred_title_terms=tuple(title_terms[:self._max_title_terms]),
                prefer_newest=prefer_newest,
                confidence=confidence,
                reason=reason,
            )

            logger.info(
                "metadata_intent.parsed has_pref=%s confidence=%.2f signals=%d",
                pref.has_preferences, pref.confidence, signals,
            )

            return pref

        except Exception:
            logger.warning("metadata_intent.parse_error", exc_info=True)
            return MetadataPreference.empty()

    # ── Detection helpers ────────────────────────────────────────────

    def _detect_representation_type(self, q: str) -> list[str]:
        """Detect if query prefers original or synthesized content."""
        types: list[str] = []
        if any(kw in q for kw in _SUMMARY_KW):
            types.append("synthesized")
        if any(kw in q for kw in _ORIGINAL_KW):
            types.append("original")
        # If both detected, ambiguous → no preference
        if len(types) > 1:
            return []
        return types

    def _detect_source_category(self, q: str) -> list[str]:
        """Detect source/category preferences from keywords."""
        sources: list[str] = []
        for keywords, label in _SOURCE_KEYWORD_MAP:
            if any(kw in q for kw in keywords):
                sources.append(label)
        return sources[:2]  # max 2 categories

    def _extract_title_terms(self, q: str) -> list[str]:
        """
        Extract potential title-matching terms from query.

        Heuristic: words that are likely document names/titles
        (capitalized words, quoted phrases, or specific nouns).
        """
        terms: list[str] = []

        # Extract quoted phrases
        quoted = re.findall(r'"([^"]+)"', q) + re.findall(r"'([^']+)'", q)
        for phrase in quoted:
            phrase = phrase.strip()
            if len(phrase) >= 2:
                terms.append(phrase)

        return terms[:self._max_title_terms]

    @staticmethod
    def _detect_recency(q: str) -> bool:
        """Detect if query prefers newest/latest documents."""
        return any(kw in q for kw in _RECENCY_KW)
