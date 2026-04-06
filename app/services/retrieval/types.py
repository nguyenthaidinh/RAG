"""
Shared domain types for the retrieval engine.

Every component in the retrieval package exchanges data through these
frozen DTOs — no mutable state, no ORM coupling.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


def make_chunk_id(document_id: int, chunk_index: int) -> int:
    """Synthetic chunk ID: deterministic, reversible."""
    return document_id * 100_000 + chunk_index


@dataclass(frozen=True)
class VectorFilter:
    """Constrains vector search to an allowed set of documents."""

    document_ids: frozenset[int]         # allowed docs from AccessPolicy
    document_id: int | None = None       # optional narrowing
    version_id: str | None = None        # optional narrowing


@dataclass(frozen=True)
class QueryScope:
    """Scope for the retrieval query (extensible for folders/projects)."""

    mode: Literal["all"] = "all"


@dataclass(frozen=True)
class ScoredChunk:
    """A chunk with a relevance score from a retrieval source."""

    chunk_id: int
    document_id: int
    version_id: str
    chunk_index: int
    score: float
    source: Literal["vector", "bm25", "hybrid", "rerank"]
    snippet: str
    title: str | None = None
    heading: str | None = None


@dataclass(frozen=True)
class QueryResult:
    """Final result returned to the API consumer."""

    chunk_id: int
    document_id: int
    score: float
    snippet: str
    highlights: tuple[str, ...]  # immutable sequence of highlighted terms

    # ── Document metadata (Phase 1 hotfix) ────────────────────────
    title: str | None = None
    heading: str | None = None

    # ── Source fidelity (Step 5) ──────────────────────────────────
    # Always points to the original document, even if selected chunk
    # came from a synthesized child.  None for pre-synthesis docs.
    source_document_id: int | None = None

    # ── Debug metadata (Step 5) ──────────────────────────────────
    # Only populated when debug mode is active.  None by default
    # to keep the public contract unchanged.
    debug_meta: dict | None = None

