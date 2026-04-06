"""
Canonical knowledge item — unified data model for Source Platform.

Every source connector maps its raw data into a CanonicalKnowledgeItem
before handing it to SourceSyncService → DocumentService.upsert().

This keeps the document pipeline agnostic of upstream data formats.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


# ── Normalisation helpers ────────────────────────────────────────────

_MULTI_WHITESPACE = re.compile(r"[ \t]+")
_MULTI_NEWLINES = re.compile(r"\n{3,}")


def _normalise_text(text: str) -> str:
    """Collapse redundant whitespace without destroying paragraph breaks."""
    text = text.strip()
    text = _MULTI_WHITESPACE.sub(" ", text)
    text = _MULTI_NEWLINES.sub("\n\n", text)
    return text


# ── Canonical model ──────────────────────────────────────────────────


@dataclass
class CanonicalKnowledgeItem:
    """
    Unified representation of a knowledge item from any source.

    Fields
    ------
    external_id : str
        Unique identifier within the source system.
    source_key : str
        Short key identifying the source definition, e.g. ``"core-api"``.
    source_type : str
        Connector type, e.g. ``"internal_api"``, ``"database"``, ``"html"``.
    title : str | None
        Human-readable title.
    body_text : str
        Main textual content that will be ingested.
    summary : str | None
        Optional short summary / excerpt.
    source_uri : str | None
        URI pointing back to the original item in the source system.
    updated_at : datetime | None
        Last-modified timestamp from the source (if available).
    checksum : str | None
        Pre-computed checksum from the source.  When absent,
        ``effective_checksum()`` derives one deterministically.
    metadata : dict[str, Any]
        Arbitrary key/value pairs from the source system.
    access_scope : dict[str, Any]
        Access-control hints (tenant, roles, visibility, etc.).
    representation_type : str
        Document representation type passed to DocumentService
        (default ``"original"``).
    """

    external_id: str
    source_key: str
    source_type: str
    title: str | None
    body_text: str

    summary: str | None = None
    source_uri: str | None = None
    updated_at: datetime | None = None
    checksum: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    access_scope: dict[str, Any] = field(default_factory=dict)
    representation_type: str = "original"

    # ── Normalisation ────────────────────────────────────────────

    def normalized_title(self) -> str | None:
        """Return whitespace-normalised title, or None if empty."""
        if not self.title:
            return None
        cleaned = self.title.strip()
        return cleaned or None

    def normalized_body_text(self) -> str:
        """Return whitespace-normalised body text."""
        return _normalise_text(self.body_text) if self.body_text else ""

    # ── Checksum ─────────────────────────────────────────────────

    def effective_checksum(self) -> str:
        """
        Return a stable content checksum.

        If the source already provides a ``checksum``, use it directly.
        Otherwise, derive a deterministic SHA-256 from the item's core
        identity + content fields.
        """
        if self.checksum:
            return self.checksum

        # Build a deterministic payload from identity + content fields.
        payload = {
            "external_id": self.external_id,
            "source_key": self.source_key,
            "title": self.normalized_title() or "",
            "body_text": self.normalized_body_text(),
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
