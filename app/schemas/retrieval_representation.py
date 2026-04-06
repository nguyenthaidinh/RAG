"""
Representation intent schemas (Phase 3D).

Defines RepresentationIntent and RepresentationPreference.
Used by RepresentationIntentService to communicate to the
family consolidation / representation selector.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RepresentationIntent(str, Enum):
    """Classification of query's representation need."""

    OVERVIEW_SUMMARY = "overview_summary"
    EXACT_SPECIFIC = "exact_specific"
    CITATION_SENSITIVE = "citation_sensitive"
    EXPLANATORY_MIXED = "explanatory_mixed"
    BALANCED_DEFAULT = "balanced_default"


@dataclass(frozen=True)
class RepresentationPreference:
    """
    Safe representation preference for family selection.

    Fields:
      - intent: classified intent type
      - preferred_type: 'original' | 'synthesized' | 'balanced'
      - strength: 0.0..1.0 how strongly to apply the preference
      - confidence: 0.0..1.0 classification confidence
      - reason: human-readable explanation
    """

    intent: RepresentationIntent = RepresentationIntent.BALANCED_DEFAULT
    preferred_type: str = "balanced"  # 'original' | 'synthesized' | 'balanced'
    strength: float = 0.0
    confidence: float = 0.0
    reason: str = ""

    @property
    def has_preference(self) -> bool:
        """True if a non-balanced preference is set with sufficient strength."""
        return self.preferred_type != "balanced" and self.strength > 0.01

    def telemetry_dict(self) -> dict:
        """Safe telemetry — no raw text."""
        return {
            "intent": self.intent.value,
            "preferred_type": self.preferred_type,
            "strength": round(self.strength, 3),
            "confidence": round(self.confidence, 3),
            "has_preference": self.has_preference,
        }

    @classmethod
    def neutral(cls) -> "RepresentationPreference":
        """No preference — passthrough."""
        return cls(
            intent=RepresentationIntent.BALANCED_DEFAULT,
            preferred_type="balanced",
            strength=0.0,
            confidence=0.0,
            reason="no_representation_intent",
        )
