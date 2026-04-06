from __future__ import annotations

from pydantic import BaseModel, Field


class MetadataSearchConditions(BaseModel):
    document_kinds: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    source_labels: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    audience: list[str] = Field(default_factory=list)
    title_terms: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    freshness_sensitive: bool = False


class MetadataCandidate(BaseModel):
    document_id: int
    title: str | None = None
    source: str
    external_id: str
    representation_type: str
    parent_document_id: int | None = None
    metadata: dict = Field(default_factory=dict)
    content_text: str | None = None
    score: float = 0.0
    reasons: list[str] = Field(default_factory=list)