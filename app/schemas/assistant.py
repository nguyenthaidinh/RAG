"""
Assistant API schemas — generic, platform-agnostic.

These schemas define the contract for the high-level assistant endpoint.
They are intentionally decoupled from the low-level query/retrieval schemas
so that the assistant API can evolve independently.

Design principles:
  - Generic: no app-specific fields (no URLs, no business routes)
  - Stable: field names are final (message, citations, actions)
  - Safe defaults: lists default to [], optional fields to None

Phase 1 additions:
  - AssistantConversationTurn: conversation history support
  - AssistantCitation: source_document_id, heading, rank for citation fidelity
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────────────────────
# Request models
# ──────────────────────────────────────────────────────────────────────────────


class AssistantRetrievalOptions(BaseModel):
    """Optional tuning knobs for the retrieval layer."""

    mode: Literal["hybrid", "vector", "bm25"] | None = Field(
        default=None,
        description="Retrieval mode. Defaults to system default if omitted.",
    )
    final_limit: int | None = Field(
        default=None, ge=1, le=100,
        description="Max results returned after re-ranking.",
    )
    top_k: int | None = Field(
        default=None, ge=1, le=100,
        description="Alias for final_limit (legacy compat).",
    )
    vector_limit: int | None = Field(
        default=None, ge=1, le=500,
        description="Max candidates from vector search.",
    )
    bm25_limit: int | None = Field(
        default=None, ge=1, le=500,
        description="Max candidates from BM25 search.",
    )


class AssistantContext(BaseModel):
    """
    Optional client-side context.

    Generic metadata that any consumer app can pass.
    AI Server does NOT interpret these fields — they are stored/forwarded
    for traceability and future use (e.g., context-aware ranking).
    """

    current_module: str | None = Field(
        default=None, max_length=256,
        description="Module/page the user is currently on.",
    )
    current_url: str | None = Field(
        default=None, max_length=2048,
        description="Current URL in the consumer app.",
    )
    roles: list[str] | None = Field(
        default=None,
        description="User roles in the consumer app (informational).",
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Arbitrary key-value metadata from the consumer.",
    )


class AssistantConversationTurn(BaseModel):
    """A single turn in the conversation history.

    Defined locally (not imported from query schemas) so the assistant
    API contract can evolve independently of the retrieval API.
    """

    role: Literal["user", "assistant"] = Field(
        ..., description="Who sent this turn.",
    )
    text: str = Field(
        ..., max_length=2000,
        description="Content of this turn, trimmed to 2000 chars max.",
    )


class AssistantRespondRequest(BaseModel):
    """Request body for POST /api/v1/assistant/respond."""

    app_id: str | None = Field(
        default=None, max_length=128,
        description="Consumer application identifier (for analytics).",
    )
    session_id: str | None = Field(
        default=None, max_length=256,
        description="Conversation/session ID on the consumer side.",
    )
    message: str = Field(
        ..., min_length=1, max_length=8000,
        description="User message / question.",
    )
    history: list[AssistantConversationTurn] = Field(
        default_factory=list,
        description="Recent conversation history for follow-up context. "
                    "Ordered chronologically (oldest first). "
                    "Empty list if no history. Max ~8 recent turns recommended.",
    )
    context: AssistantContext | None = Field(
        default=None,
        description="Optional client-side context.",
    )
    retrieval: AssistantRetrievalOptions | None = Field(
        default=None,
        description="Optional retrieval tuning parameters.",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Response models
# ──────────────────────────────────────────────────────────────────────────────


class AssistantCitation(BaseModel):
    """A single citation tied to evidence used in answer generation.

    Phase 1: citations map to evidences that were actually used to
    produce the answer, not the full top-k retrieval set.
    """

    chunk_id: int | None = Field(
        default=None,
        description="Chunk identifier.",
    )
    document_id: int | None = Field(
        default=None,
        description="Document identifier (may be synthesized child).",
    )
    source_document_id: int | None = Field(
        default=None,
        description="Original/parent document ID for citation fidelity. "
                    "When the retrieved chunk comes from a synthesized document, "
                    "this points to the original source. None for non-synthesized docs.",
    )
    title: str | None = Field(
        default=None,
        description="Document or chunk title.",
    )
    heading: str | None = Field(
        default=None,
        description="Section heading of the cited chunk. "
                    "Useful for section-level citation display.",
    )
    snippet: str | None = Field(
        default=None,
        description="Relevant text excerpt used as evidence.",
    )
    score: float | None = Field(
        default=None,
        description="Relevance score from retrieval.",
    )
    rank: int | None = Field(
        default=None,
        description="Evidence rank (1 = most relevant). "
                    "Rank is assigned during evidence selection, "
                    "before answer generation.",
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Additional metadata about this citation.",
    )


class AssistantAction(BaseModel):
    """
    A suggested action for the consumer app.

    V1: always empty list. Structure reserved for future phases.
    Consumer apps interpret target_key / action type on their side.
    AI Server never returns hard-coded URLs or business routes.
    """

    type: str = Field(..., description="Action type identifier.")
    target_key: str | None = Field(
        default=None,
        description="Generic key the consumer app can resolve.",
    )
    label: str | None = Field(
        default=None,
        description="Human-readable label for the action.",
    )
    payload: dict[str, Any] | None = Field(
        default=None,
        description="Arbitrary payload for the consumer app.",
    )


class AssistantRespondResponse(BaseModel):
    """Response body for POST /api/v1/assistant/respond."""

    message: str = Field(
        ...,
        description="Assistant response message.",
    )
    citations: list[AssistantCitation] = Field(
        default_factory=list,
        description="Source citations from retrieval. Always a list (never null).",
    )
    actions: list[AssistantAction] = Field(
        default_factory=list,
        description="Suggested actions. Always a list (never null). V1: empty.",
    )
    trace_id: str | None = Field(
        default=None,
        description="Request trace ID for debugging/correlation.",
    )
