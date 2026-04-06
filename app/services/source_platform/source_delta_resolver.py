"""
Source delta resolver (Phase 8 — Delta-Aware Sync).

Pure logic class — no DB access, no I/O.
Compares an existing link (or None) against a canonical item
to determine the required sync action.

Actions::

    CREATE       — no existing link → new item
    UPDATE       — link exists, checksum changed
    SKIP         — link exists, checksum same, status=active
    REACTIVATE   — link was missing, item came back
"""
from __future__ import annotations

import enum
import logging
from dataclasses import dataclass

from app.db.models.source_document_link import SourceDocumentLink
from app.services.source_platform.canonical_item import CanonicalKnowledgeItem

logger = logging.getLogger(__name__)


class DeltaAction(enum.Enum):
    """Sync action determined by delta resolution."""

    CREATE = "create"
    UPDATE = "update"
    SKIP = "skip"
    REACTIVATE = "reactivate"


@dataclass(frozen=True)
class DeltaResult:
    """Result of delta resolution for one item.

    Attributes
    ----------
    action : DeltaAction
        What to do with this item.
    existing_link : SourceDocumentLink | None
        The existing link record, if any.
    reason : str
        Short human-readable explanation for logging.
    """

    action: DeltaAction
    existing_link: SourceDocumentLink | None
    reason: str


class SourceDeltaResolver:
    """Stateless resolver: (existing_link, canonical_item) → DeltaResult.

    Decision table:
    ┌────────────────────────┬──────────────┬────────────────────────┐
    │ existing_link          │ checksum     │ action                 │
    ├────────────────────────┼──────────────┼────────────────────────┤
    │ None                   │ —            │ CREATE                 │
    │ status='missing'       │ —            │ REACTIVATE             │
    │ status='error'         │ —            │ UPDATE (retry)         │
    │ status='active'        │ changed      │ UPDATE                 │
    │ status='active'        │ same         │ SKIP                   │
    └────────────────────────┴──────────────┴────────────────────────┘
    """

    __slots__ = ()

    def resolve(
        self,
        *,
        existing_link: SourceDocumentLink | None,
        canonical: CanonicalKnowledgeItem,
    ) -> DeltaResult:
        """Determine sync action for one item."""

        new_checksum = canonical.effective_checksum()

        # ── No existing link → CREATE ────────────────────────────
        if existing_link is None:
            return DeltaResult(
                action=DeltaAction.CREATE,
                existing_link=None,
                reason="new_item",
            )

        # ── Missing → REACTIVATE (always re-ingest) ─────────────
        if existing_link.status == "missing":
            return DeltaResult(
                action=DeltaAction.REACTIVATE,
                existing_link=existing_link,
                reason="reactivate_missing",
            )

        # ── Error → UPDATE (retry ingest) ────────────────────────
        if existing_link.status == "error":
            return DeltaResult(
                action=DeltaAction.UPDATE,
                existing_link=existing_link,
                reason="retry_error",
            )

        # ── Active: compare checksum ─────────────────────────────
        if existing_link.content_checksum != new_checksum:
            return DeltaResult(
                action=DeltaAction.UPDATE,
                existing_link=existing_link,
                reason="checksum_changed",
            )

        return DeltaResult(
            action=DeltaAction.SKIP,
            existing_link=existing_link,
            reason="unchanged",
        )
