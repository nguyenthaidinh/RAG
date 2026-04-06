"""
Metadata intent schemas (Phase 3B).

Defines MetadataPreference — the output of MetadataIntentService.
All fields are safe whitelist-only values derived from actual schema.
"""
from __future__ import annotations

from dataclasses import dataclass, field


# ── Whitelist of supported metadata fields ────────────────────────────
# Only these fields are allowed in metadata preferences.
# Maps to actual columns/JSONB keys in the documents table.

SUPPORTED_FIELDS = frozenset({
    "source",                # documents.source (String 50)
    "title",                 # documents.title (String 512)
    "representation_type",   # documents.representation_type (original|synthesized)
    "file_name",             # documents.meta -> file_name
    "content_type",          # documents.meta -> content_type
})


@dataclass(frozen=True)
class MetadataPreference:
    """
    Safe, whitelist-only metadata preference for retrieval bias.

    Fields:
      - preferred_sources: match against documents.source
      - preferred_representation_types: 'original' | 'synthesized'
      - preferred_title_terms: keywords to match against documents.title
      - prefer_newest: bias toward newer documents
      - confidence: 0.0..1.0
      - reason: human-readable explanation
    """

    preferred_sources: tuple[str, ...] = ()
    preferred_representation_types: tuple[str, ...] = ()
    preferred_title_terms: tuple[str, ...] = ()
    prefer_newest: bool = False
    confidence: float = 0.0
    reason: str = ""

    @property
    def has_preferences(self) -> bool:
        """True if any non-empty preference is set."""
        return bool(
            self.preferred_sources
            or self.preferred_representation_types
            or self.preferred_title_terms
            or self.prefer_newest
        )

    def telemetry_dict(self) -> dict:
        """Safe telemetry — no raw terms longer than needed."""
        return {
            "has_preferences": self.has_preferences,
            "source_count": len(self.preferred_sources),
            "repr_type_count": len(self.preferred_representation_types),
            "title_term_count": len(self.preferred_title_terms),
            "prefer_newest": self.prefer_newest,
            "confidence": round(self.confidence, 3),
        }

    @classmethod
    def empty(cls) -> "MetadataPreference":
        """No preference — passthrough."""
        return cls(confidence=0.0, reason="no_metadata_intent")
