"""
Metadata-aware reranker wrapper (Phase 3B).

Wraps the existing DeterministicReRanker and applies soft metadata
bias AFTER the base reranking score is computed.

Design:
  - Additive bias only — never removes results
  - Bias weights are configurable and small by default
  - Requires document metadata lookup (source, representation_type, title)
  - Fail-open: if metadata lookup fails, returns base reranked results
  - Feature-flagged via METADATA_RETRIEVAL_ENABLED
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from app.core.config import settings
from app.schemas.retrieval_metadata import MetadataPreference
from app.services.retrieval.types import ScoredChunk

logger = logging.getLogger(__name__)


class MetadataBiasReranker:
    """
    Applies soft metadata bias to reranked chunks.

    Does NOT replace the base reranker — operates after it.
    Receives document metadata and MetadataPreference, adds small
    score adjustments, and re-sorts.
    """

    __slots__ = (
        "_source_weight",
        "_repr_weight",
        "_title_weight",
        "_recency_weight",
    )

    def __init__(self) -> None:
        self._source_weight = float(
            getattr(settings, "METADATA_RETRIEVAL_SOURCE_BIAS_WEIGHT", 0.08)
        )
        self._repr_weight = float(
            getattr(settings, "METADATA_RETRIEVAL_REPRESENTATION_BIAS_WEIGHT", 0.10)
        )
        self._title_weight = float(
            getattr(settings, "METADATA_RETRIEVAL_TITLE_BIAS_WEIGHT", 0.06)
        )
        self._recency_weight = float(
            getattr(settings, "METADATA_RETRIEVAL_RECENCY_BIAS_WEIGHT", 0.05)
        )

    def apply_bias(
        self,
        *,
        chunks: list[ScoredChunk],
        preference: MetadataPreference,
        doc_metadata: dict[int, dict],
    ) -> list[ScoredChunk]:
        """
        Apply metadata bias to chunks and re-sort.

        Args:
            chunks: Already reranked chunks from base reranker.
            preference: Metadata preferences from MetadataIntentService.
            doc_metadata: {doc_id: {source, title, representation_type,
                          created_at, meta}} from DB.

        Returns:
            Re-sorted chunks with metadata bias applied.
        """
        if not chunks or not preference.has_preferences:
            return chunks

        biased: list[ScoredChunk] = []
        bias_applied_count = 0

        for chunk in chunks:
            meta = doc_metadata.get(chunk.document_id, {})
            bias = self._compute_bias(preference, meta)

            if abs(bias) > 0.001:
                bias_applied_count += 1
                new_score = round(chunk.score + bias, 8)
                biased.append(
                    ScoredChunk(
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        version_id=chunk.version_id,
                        chunk_index=chunk.chunk_index,
                        score=new_score,
                        source=chunk.source,
                        snippet=chunk.snippet,
                        title=chunk.title,
                        heading=chunk.heading,
                    )
                )
            else:
                biased.append(chunk)

        # Re-sort by score descending
        biased.sort(key=lambda c: (-c.score, c.chunk_id))

        logger.info(
            "metadata_bias.applied total=%d biased=%d "
            "source_w=%.3f repr_w=%.3f title_w=%.3f recency_w=%.3f",
            len(chunks), bias_applied_count,
            self._source_weight, self._repr_weight,
            self._title_weight, self._recency_weight,
        )

        return biased

    def _compute_bias(
        self,
        pref: MetadataPreference,
        meta: dict,
    ) -> float:
        """Compute total bias for a single document."""
        bias = 0.0

        # 1) Source/category match
        if pref.preferred_sources and self._source_weight > 0:
            doc_source = (meta.get("source") or "").lower()
            doc_meta_json = meta.get("meta") or {}
            doc_category = (doc_meta_json.get("category") or "").lower() if isinstance(doc_meta_json, dict) else ""
            doc_file_name = (doc_meta_json.get("file_name") or "").lower() if isinstance(doc_meta_json, dict) else ""

            for pref_source in pref.preferred_sources:
                ps = pref_source.lower()
                if ps in doc_source or ps in doc_category or ps in doc_file_name:
                    bias += self._source_weight
                    break

        # 2) Representation type match
        if pref.preferred_representation_types and self._repr_weight > 0:
            doc_repr = (meta.get("representation_type") or "original").lower()
            if doc_repr in [r.lower() for r in pref.preferred_representation_types]:
                bias += self._repr_weight

        # 3) Title term match
        if pref.preferred_title_terms and self._title_weight > 0:
            doc_title = (meta.get("title") or "").lower()
            if doc_title:
                matched_terms = sum(
                    1 for term in pref.preferred_title_terms
                    if term.lower() in doc_title
                )
                if matched_terms > 0:
                    # Proportional boost
                    ratio = min(matched_terms / len(pref.preferred_title_terms), 1.0)
                    bias += self._title_weight * ratio

        # 4) Recency preference
        if pref.prefer_newest and self._recency_weight > 0:
            created = meta.get("created_at")
            if created and isinstance(created, datetime):
                now = datetime.now(timezone.utc)
                age_days = (now - created).days if created.tzinfo else 365
                if age_days <= 30:
                    bias += self._recency_weight
                elif age_days <= 90:
                    bias += self._recency_weight * 0.5
                # older than 90 days: no boost

        return bias
