"""
Document lifecycle state machine.

States::

    uploaded ──► chunked ──► indexed ──► ready
        │            │           │          │
        └──► error ◄─┘───────◄──┘────────◄─┘

- **uploaded**: content received, not yet processed.
- **chunked**: NLP clean + chunk complete, awaiting vector indexing.
- **indexed**: vectors stored in index, awaiting final validation.
- **ready**: fully processed, available for retrieval / RAG.
- **error**: processing failed at any stage (retryable).

Legacy aliases (``pending``, ``processing``) are accepted in the
transition table for backward compatibility with Phase 1/2 data.
"""
from __future__ import annotations

# ── canonical states ──────────────────────────────────────────────────

UPLOADED: str = "uploaded"
CHUNKED: str = "chunked"
INDEXED: str = "indexed"
READY: str = "ready"
ERROR: str = "error"

# legacy aliases (still valid in DB, mapped for transition checks)
PENDING: str = "pending"       # ≡ uploaded
PROCESSING: str = "processing"  # ≡ chunked

ALL_STATES: frozenset[str] = frozenset({
    UPLOADED, CHUNKED, INDEXED, READY, ERROR,
    PENDING, PROCESSING,
})

# ── transition rules ─────────────────────────────────────────────────

_TRANSITIONS: dict[str, frozenset[str]] = {
    UPLOADED:    frozenset({CHUNKED, ERROR}),
    PENDING:     frozenset({CHUNKED, ERROR}),            # legacy alias
    CHUNKED:     frozenset({CHUNKED, INDEXED, READY, ERROR}),  # self: re-ingest before indexed
    PROCESSING:  frozenset({CHUNKED, INDEXED, READY, ERROR}),  # legacy alias
    INDEXED:     frozenset({CHUNKED, READY, ERROR}),     # re-ingest after indexed
    READY:       frozenset({CHUNKED, ERROR}),             # re-ingest
    ERROR:       frozenset({UPLOADED, CHUNKED}),          # retry
}


class InvalidTransitionError(Exception):
    """Raised when a document status transition is not allowed."""


def validate_transition(current: str, target: str) -> None:
    """
    Raise ``InvalidTransitionError`` if *current* → *target* is not a
    valid lifecycle transition.
    """
    allowed = _TRANSITIONS.get(current)
    if allowed is None:
        raise InvalidTransitionError(
            f"Unknown document state '{current}'"
        )
    if target not in allowed:
        raise InvalidTransitionError(
            f"Cannot transition from '{current}' to '{target}'. "
            f"Allowed: {sorted(allowed)}"
        )
