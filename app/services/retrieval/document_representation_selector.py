"""
Document representation selector (Step 4 + Step 5 hardening).

Consolidates retrieval candidates from dual-document families
(original + synthesized) so that only **one representative per family**
enters the final LLM context.

Design:
- Lightweight query-intent heuristic (summary / specific / neutral).
- Family grouping via ``parent_document_id`` linkage.
- Configurable policy mode (balanced / summary_first / source_first).
- Deterministic, testable, no ML dependency.
- Fully backward compatible: documents without a synthesized child
  pass through unchanged.

Usage::

    selector = DocumentRepresentationSelector(mode="balanced")
    consolidated = selector.consolidate(query_text, candidates)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from enum import Enum
from typing import Literal, Sequence

logger = logging.getLogger(__name__)


# ── Policy modes ─────────────────────────────────────────────────────

RepresentationMode = Literal["balanced", "summary_first", "source_first"]

_VALID_MODES: frozenset[str] = frozenset({"balanced", "summary_first", "source_first"})


# ── Score thresholds (easy to tune) ──────────────────────────────────

CLOSE_SCORE_DELTA = 0.03
"""Scores within this delta are considered "close" — tie-breaking applies."""

CLEAR_WIN_DELTA = 0.08
"""Score gap above this means one candidate is a clear winner."""

# Mode-specific overrides for CLOSE_SCORE_DELTA (how aggressively to
# favor the preferred type when scores are "close").
_MODE_CLOSE_DELTA: dict[str, float] = {
    "balanced": 0.03,
    "summary_first": 0.06,   # wider tie zone → synthesized wins more
    "source_first": 0.06,    # wider tie zone → original wins more
}


# ── Query intent ─────────────────────────────────────────────────────

class IntentType(str, Enum):
    SUMMARY = "summary"
    SPECIFIC = "specific"
    NEUTRAL = "neutral"


_SUMMARY_KEYWORDS: frozenset[str] = frozenset({
    # Vietnamese
    "tóm tắt", "tổng quan", "nội dung chính", "điểm chính",
    "khái quát", "tổng hợp",
    # English
    "overview", "summary", "summarize", "summarise",
    "key points", "main points", "highlights", "gist",
})

_SPECIFIC_KEYWORDS: frozenset[str] = frozenset({
    # Vietnamese
    "điều", "khoản", "mục", "chương", "chi tiết", "cụ thể",
    "chính xác", "nguyên văn", "theo quy định", "bước nào",
    "quy định", "điều khoản",
    # English
    "clause", "section", "article", "specific", "exact",
    "verbatim", "detail", "detailed", "step by step",
    "according to", "paragraph",
})


def detect_intent(query: str) -> IntentType:
    """
    Lightweight keyword-based intent detection.

    Checks if the lowercased query contains any known summary or specific
    keywords.  Multi-word keywords are matched as substrings.

    Returns ``IntentType.NEUTRAL`` when neither signal is strong enough.
    """
    q = query.lower()

    summary_score = sum(1 for kw in _SUMMARY_KEYWORDS if kw in q)
    specific_score = sum(1 for kw in _SPECIFIC_KEYWORDS if kw in q)

    if summary_score > specific_score:
        return IntentType.SUMMARY
    if specific_score > summary_score:
        return IntentType.SPECIFIC
    if summary_score > 0 and summary_score == specific_score:
        # Ambiguous → neutral
        return IntentType.NEUTRAL

    return IntentType.NEUTRAL


# ── Candidate / Selected DTOs ────────────────────────────────────────

@dataclass(frozen=True)
class RetrievalCandidate:
    """
    Enriched retrieval hit carrying document-family metadata.

    Built from a ``ScoredChunk`` + document-table lookups.
    """
    document_id: int
    chunk_id: int
    chunk_index: int
    score: float
    snippet: str
    title: str | None
    version_id: str
    source: str                          # "vector" | "bm25" | "hybrid" | "rerank"
    representation_type: str             # "original" | "synthesized"
    parent_document_id: int | None       # None for originals


@dataclass(frozen=True)
class SelectedCandidate:
    """
    A candidate chosen as the family representative, annotated with
    debug metadata and source fidelity info.
    """
    document_id: int
    chunk_id: int
    chunk_index: int
    score: float
    snippet: str
    title: str | None
    version_id: str
    source: str

    # ── debug / logging metadata ─────────────────────────────────
    family_key: int
    selected_representation_type: str
    selection_reason: str

    # ── source fidelity (Step 5) ─────────────────────────────────
    source_document_id: int
    """
    Always points to the original document for citation/source purposes.
    - If selected is original → source_document_id == document_id
    - If selected is synthesized → source_document_id == parent_document_id
    """


# ── Selector service ─────────────────────────────────────────────────

def _family_key(candidate: RetrievalCandidate) -> int:
    """
    Derive the family key for grouping.

    - original  → family_key = document_id
    - synthesized with parent → family_key = parent_document_id
    - synthesized without parent (shouldn't happen) → document_id
    """
    if candidate.representation_type == "synthesized" and candidate.parent_document_id is not None:
        return candidate.parent_document_id
    return candidate.document_id


def _source_document_id(candidate: RetrievalCandidate) -> int:
    """
    Resolve the original/source document ID for citation fidelity.

    - original → source = self
    - synthesized → source = parent (fallback self if no parent)
    """
    if candidate.representation_type == "synthesized" and candidate.parent_document_id is not None:
        return candidate.parent_document_id
    return candidate.document_id


class DocumentRepresentationSelector:
    """
    Consolidates retrieval candidates so that each document family
    contributes at most **one** representative to the final context.

    Accepts a ``mode`` that adjusts tie-breaking behavior:
    - ``balanced``: default Step 4 behavior
    - ``summary_first``: wider tie zone favoring synthesized
    - ``source_first``: wider tie zone favoring original
    """

    __slots__ = ("_mode", "_close_delta")

    def __init__(self, mode: RepresentationMode = "balanced") -> None:
        m = mode.strip().lower() if isinstance(mode, str) else "balanced"
        if m not in _VALID_MODES:
            logger.warning(
                "representation_selector.invalid_mode mode=%s falling_back=balanced", m,
            )
            m = "balanced"
        self._mode: str = m
        self._close_delta: float = _MODE_CLOSE_DELTA.get(m, CLOSE_SCORE_DELTA)

    @property
    def mode(self) -> str:
        return self._mode

    @staticmethod
    def detect_intent(query: str) -> IntentType:
        """Delegate to module-level heuristic for testability."""
        return detect_intent(query)

    @staticmethod
    def family_key(candidate: RetrievalCandidate) -> int:
        """Delegate to module-level helper."""
        return _family_key(candidate)

    def consolidate(
        self,
        query: str,
        candidates: Sequence[RetrievalCandidate],
        *,
        representation_preference=None,
    ) -> list[SelectedCandidate]:
        """
        Group *candidates* by family, pick one representative per family,
        and return a score-sorted list of ``SelectedCandidate``.

        If a family has only one member, it is selected unconditionally.

        Args:
            representation_preference: Optional Phase 3D preference.
                When present with has_preference, applies a soft score
                boost to preferred-type candidates before selection.
        """
        if not candidates:
            return []

        intent = self.detect_intent(query)

        # ── Phase 3D: Apply representation preference boost ──────
        # Adjust scores before family selection so tie-breaking
        # naturally favors the preferred type.
        adjusted_candidates = list(candidates)
        repr_boost_applied = False
        if (
            representation_preference is not None
            and getattr(representation_preference, "has_preference", False)
        ):
            pref_type = representation_preference.preferred_type
            strength = representation_preference.strength
            if pref_type in ("original", "synthesized") and strength > 0:
                adjusted_candidates = [
                    RetrievalCandidate(
                        document_id=c.document_id,
                        chunk_id=c.chunk_id,
                        chunk_index=c.chunk_index,
                        score=round(c.score + strength, 8) if c.representation_type == pref_type else c.score,
                        snippet=c.snippet,
                        title=c.title,
                        version_id=c.version_id,
                        source=c.source,
                        representation_type=c.representation_type,
                        parent_document_id=c.parent_document_id,
                    )
                    for c in candidates
                ]
                repr_boost_applied = True
                logger.info(
                    "representation_selector.policy_boost type=%s strength=%.3f intent=%s",
                    pref_type, strength, representation_preference.intent.value,
                )

        # ── Group by family ──────────────────────────────────────
        families: dict[int, list[RetrievalCandidate]] = {}
        for c in adjusted_candidates:
            fk = self.family_key(c)
            families.setdefault(fk, []).append(c)

        # ── Choose best per family ───────────────────────────────
        selected: list[SelectedCandidate] = []
        for fk, members in families.items():
            winner = self._choose_within_family(intent, members)
            selected.append(winner)

        # ── Sort by score descending, stable ─────────────────────
        selected.sort(key=lambda s: (-s.score, s.document_id))

        logger.debug(
            "representation_selector.consolidate intent=%s mode=%s "
            "input_candidates=%d families=%d output=%d policy_boost=%s",
            intent.value, self._mode,
            len(candidates), len(families), len(selected),
            repr_boost_applied,
        )

        return selected

    # ── internal ─────────────────────────────────────────────────

    def _choose_within_family(
        self,
        intent: IntentType,
        members: list[RetrievalCandidate],
    ) -> SelectedCandidate:
        """
        Pick the best candidate within a single family.

        If only one member exists, it wins by default.
        If two members exist (original + synthesized), apply intent-aware
        tie-breaking.
        """
        if len(members) == 1:
            c = members[0]
            reason = f"fallback_{c.representation_type}_only"
            return self._to_selected(c, reason)

        # Separate by type
        originals = [m for m in members if m.representation_type == "original"]
        synthesized = [m for m in members if m.representation_type == "synthesized"]

        # Best of each type (highest score)
        best_original = max(originals, key=lambda c: c.score) if originals else None
        best_synth = max(synthesized, key=lambda c: c.score) if synthesized else None

        # Only one type present
        if best_original and not best_synth:
            return self._to_selected(best_original, "fallback_original_only")
        if best_synth and not best_original:
            return self._to_selected(best_synth, "fallback_synthesized_only")

        # Both present — apply intent + score + mode logic
        assert best_original is not None and best_synth is not None
        return self._pick_by_intent(intent, best_original, best_synth)

    def _pick_by_intent(
        self,
        intent: IntentType,
        original: RetrievalCandidate,
        synthesized: RetrievalCandidate,
    ) -> SelectedCandidate:
        """Apply intent-aware + mode-aware selection."""
        delta = original.score - synthesized.score
        close_delta = self._close_delta

        # ── MODE OVERRIDES ───────────────────────────────────────
        if self._mode == "summary_first":
            # Always lean synthesized unless original is a clear winner
            if delta >= CLEAR_WIN_DELTA:
                return self._to_selected(original, "summary_first_clear_win_original")
            return self._to_selected(synthesized, "summary_first_prefer_synthesized")

        if self._mode == "source_first":
            # Always lean original unless synthesized is a clear winner
            if -delta >= CLEAR_WIN_DELTA:
                return self._to_selected(synthesized, "source_first_clear_win_synthesized")
            return self._to_selected(original, "source_first_prefer_original")

        # ── BALANCED MODE (default) ──────────────────────────────

        if intent == IntentType.SUMMARY:
            # Prefer synthesized UNLESS original wins clearly
            if delta >= CLEAR_WIN_DELTA:
                return self._to_selected(
                    original, "summary_intent_clear_win_original",
                )
            return self._to_selected(
                synthesized, "summary_intent_prefer_synthesized",
            )

        if intent == IntentType.SPECIFIC:
            # Prefer original UNLESS synthesized wins clearly
            if -delta >= CLEAR_WIN_DELTA:  # synthesized score much higher
                return self._to_selected(
                    synthesized, "specific_intent_clear_win_synthesized",
                )
            return self._to_selected(
                original, "specific_intent_prefer_original",
            )

        # ── NEUTRAL ──────────────────────────────────────────────
        if abs(delta) <= close_delta:
            # Close scores → lean synthesized
            return self._to_selected(
                synthesized, "neutral_close_scores_prefer_synthesized",
            )
        if delta > 0:
            return self._to_selected(
                original, "neutral_higher_score_original",
            )
        return self._to_selected(
            synthesized, "neutral_higher_score_synthesized",
        )

    @staticmethod
    def _to_selected(
        c: RetrievalCandidate,
        reason: str,
    ) -> SelectedCandidate:
        return SelectedCandidate(
            document_id=c.document_id,
            chunk_id=c.chunk_id,
            chunk_index=c.chunk_index,
            score=c.score,
            snippet=c.snippet,
            title=c.title,
            version_id=c.version_id,
            source=c.source,
            family_key=_family_key(c),
            selected_representation_type=c.representation_type,
            selection_reason=reason,
            source_document_id=_source_document_id(c),
        )
