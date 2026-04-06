"""
Question classifier (Phase 1.1 — Hardened).

Rule-based classifier that categorizes incoming questions into one of:
  - knowledge: document/content/policy questions → existing retrieval
  - system:    stats/workflow/record questions → system context
  - access:    permission/authorization questions → access context
  - mixed:     questions combining multiple categories
  - unknown:   no clear signal detected

Phase 1.1 hardening:
  - Weighted keywords: multi-word phrases score higher than single words
  - Removed overly ambiguous single-word keywords (điều, mục, role, access)
  - MIXED requires stronger signals (min combined weight threshold)
  - Deterministic, no I/O, bilingual (Vietnamese + English)

Phase 2+ may upgrade to LLM-based classification.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class QuestionCategory(str, Enum):
    """High-level question classification."""

    KNOWLEDGE = "knowledge"
    SYSTEM = "system"
    ACCESS = "access"
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ClassificationResult:
    """Output of the question classifier."""

    category: QuestionCategory
    confidence: float  # 0.0..1.0
    matched_signals: tuple[str, ...]
    secondary_category: QuestionCategory | None = None

    def telemetry_dict(self) -> dict:
        """Safe telemetry — no raw question text."""
        return {
            "category": self.category.value,
            "confidence": round(self.confidence, 3),
            "signal_count": len(self.matched_signals),
            "secondary": (
                self.secondary_category.value
                if self.secondary_category
                else None
            ),
        }


# ── Weighted keyword dictionaries ────────────────────────────────────
# Each entry is (keyword, weight).
# Multi-word phrases get higher weight; ambiguous single words get lower.

_KNOWLEDGE_KEYWORDS: tuple[tuple[str, float], ...] = (
    # Vietnamese — high-specificity phrases
    ("tài liệu", 1.5), ("quy định", 1.5), ("quy chế", 1.5),
    ("biểu mẫu", 1.5), ("hướng dẫn", 1.2), ("chính sách", 1.5),
    ("điều khoản", 1.5), ("nội dung", 1.0), ("văn bản", 1.2),
    # Vietnamese — lower-specificity (could appear in non-knowledge context)
    ("chương", 0.5), ("khoản", 0.5),
    # English — high-specificity phrases
    ("document", 1.2), ("policy", 1.2), ("regulation", 1.5),
    ("template", 1.2), ("guideline", 1.2), ("procedure", 1.2),
    ("clause", 1.0), ("article", 0.8),
    # English — lower-specificity
    ("content", 0.5), ("section", 0.5),
)

_SYSTEM_KEYWORDS: tuple[tuple[str, float], ...] = (
    # Vietnamese — high-specificity phrases
    ("bao nhiêu", 1.5), ("tổng số", 1.5), ("tổng cộng", 1.5),
    ("thống kê", 1.5), ("số lượng", 1.2), ("tình trạng", 1.0),
    ("chờ duyệt", 1.5), ("đã duyệt", 1.5), ("đang chờ", 1.2),
    ("mới nhất", 1.0), ("gần đây", 1.0),
    # Vietnamese — lower-specificity
    ("trạng thái", 0.8), ("hồ sơ", 0.8), ("danh sách", 0.6),
    # English — high-specificity phrases
    ("how many", 1.5), ("statistics", 1.5), ("dashboard", 1.2),
    # English — medium-specificity
    ("total", 0.8), ("count", 0.8), ("pending", 1.0),
    ("workflow", 1.0), ("records", 0.8), ("recent", 0.6),
    ("latest", 0.6), ("summary", 0.6), ("report", 0.6),
    # English — lower-specificity (could also mean "HTTP status" etc.)
    ("status", 0.5),
)

_ACCESS_KEYWORDS: tuple[tuple[str, float], ...] = (
    # Vietnamese — high-specificity (multi-word phrases)
    ("tôi có quyền", 2.0), ("có quyền", 1.5), ("quyền truy cập", 1.8),
    ("xem được gì", 1.8), ("được phép", 1.5), ("ai xem được", 1.8),
    ("phân quyền", 1.5), ("ai có thể", 1.5), ("quyền hạn", 1.5),
    # English — high-specificity (multi-word phrases)
    ("my permissions", 2.0), ("can i", 1.5), ("am i allowed", 2.0),
    ("who can", 1.5), ("what can i", 1.8),
    # English — medium-specificity
    ("permission", 1.2), ("authorized", 1.2), ("authorization", 1.2),
)

# ── Thresholds ────────────────────────────────────────────────────────

# Minimum combined weight across 2+ categories to trigger MIXED
_MIXED_MIN_TOTAL_WEIGHT: float = 2.5
_MIXED_MIN_SECONDARY_WEIGHT: float = 1.0  # secondary category must be non-trivial


class QuestionClassifier:
    """
    Rule-based question classifier with weighted keywords.

    Phase 1.1: uses weighted keyword matching to reduce false positives.
    Multi-word phrases earn higher scores; ambiguous single words are
    down-weighted.  MIXED requires both categories to have meaningful
    signal strength.
    """

    __slots__ = ()

    def classify(self, question: str) -> ClassificationResult:
        """
        Classify a question into a QuestionCategory.

        Args:
            question: Raw question text (any language).

        Returns:
            ClassificationResult with category, confidence, and signals.
        """
        if not question or not question.strip():
            return ClassificationResult(
                category=QuestionCategory.UNKNOWN,
                confidence=0.0,
                matched_signals=(),
            )

        q = question.strip().lower()

        knowledge_hits = self._match_weighted(q, _KNOWLEDGE_KEYWORDS)
        system_hits = self._match_weighted(q, _SYSTEM_KEYWORDS)
        access_hits = self._match_weighted(q, _ACCESS_KEYWORDS)

        k_weight = sum(w for _, w in knowledge_hits)
        s_weight = sum(w for _, w in system_hits)
        a_weight = sum(w for _, w in access_hits)
        total_weight = k_weight + s_weight + a_weight

        # No signals at all
        if total_weight == 0:
            return ClassificationResult(
                category=QuestionCategory.UNKNOWN,
                confidence=0.2,
                matched_signals=(),
            )

        all_signals = (
            tuple(f"knowledge:{kw}" for kw, _ in knowledge_hits)
            + tuple(f"system:{kw}" for kw, _ in system_hits)
            + tuple(f"access:{kw}" for kw, _ in access_hits)
        )

        # Sort category weights descending
        scores = [
            (QuestionCategory.KNOWLEDGE, k_weight),
            (QuestionCategory.SYSTEM, s_weight),
            (QuestionCategory.ACCESS, a_weight),
        ]
        scores.sort(key=lambda x: x[1], reverse=True)

        primary_cat, primary_weight = scores[0]
        secondary_cat, secondary_weight = scores[1]

        # Count categories with meaningful weight
        cats_with_hits = sum(w > 0 for _, w in scores)

        # ── MIXED detection (hardened) ────────────────────────────────
        # Require: 2+ categories hit AND total weight sufficient
        # AND secondary category has non-trivial weight
        if (
            cats_with_hits >= 2
            and total_weight >= _MIXED_MIN_TOTAL_WEIGHT
            and secondary_weight >= _MIXED_MIN_SECONDARY_WEIGHT
        ):
            secondary = secondary_cat if secondary_weight > 0 else None
            confidence = min(0.6 + (total_weight * 0.04), 0.9)

            return ClassificationResult(
                category=QuestionCategory.MIXED,
                confidence=confidence,
                matched_signals=all_signals,
                secondary_category=secondary,
            )

        # ── Single dominant category ──────────────────────────────────
        # If secondary was too weak, primary wins
        confidence = min(0.5 + (primary_weight * 0.08), 0.95)

        return ClassificationResult(
            category=primary_cat,
            confidence=confidence,
            matched_signals=all_signals,
        )

    @staticmethod
    def _match_weighted(
        text: str,
        keywords: tuple[tuple[str, float], ...],
    ) -> list[tuple[str, float]]:
        """Return all (keyword, weight) pairs found in text.

        Longer phrases are checked first so they can't be double-counted
        by their substring components.
        """
        # Sort longest-first for greedy matching
        sorted_kws = sorted(keywords, key=lambda x: len(x[0]), reverse=True)
        matched: list[tuple[str, float]] = []
        remaining = text

        for kw, weight in sorted_kws:
            if kw in remaining:
                matched.append((kw, weight))
                # Remove matched phrase to prevent substring double-counting
                remaining = remaining.replace(kw, " ", 1)

        return matched
