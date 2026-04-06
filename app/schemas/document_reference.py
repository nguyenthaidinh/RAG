"""
Schemas for the ingest-reference endpoint.

Allows AI Server to ingest documents by reference (temporary_url from File-Service)
instead of requiring a multipart file upload.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator
from typing import Any, Literal
from app.schemas.ingest_mode import IngestMode


class DocumentReference(BaseModel):
    """Pointer to the remote file that AI Server should fetch."""

    provider: Literal["file-service"] = Field(
        ...,
        description="Reference provider (currently only 'file-service' is accepted)",
    )
    temporary_url: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="Pre-signed / temporary URL to download the file",
    )

    @field_validator("temporary_url")
    @classmethod
    def validate_url_scheme(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError("temporary_url must start with http:// or https://")
        return v


class IngestReferenceRequest(BaseModel):
    """
    Request body for POST /api/v1/documents/ingest-reference.

    Core-Platform sends this after uploading a file to File-Service.
    """

    source: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Source identifier, e.g. 'core-platform'",
    )
    external_id: str = Field(
        ...,
        min_length=1,
        max_length=512,
        description="External ID unique per tenant + source",
    )
    title: str | None = Field(
        None,
        max_length=512,
        description="Document title (optional, can be inferred from filename)",
    )
    reference: DocumentReference = Field(
        ...,
        description="Remote file reference",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata dict from the caller",
    )
    ingest_mode: IngestMode | None = Field(
        None,
        description="Optional ingest mode override: legacy | semantic",
    )


class IngestReferenceResponse(BaseModel):
    """Response for ingest-reference endpoint."""

    status: str = Field(..., description="Response status ('ok')")
    action: str = Field(..., description="Action taken: 'created', 'updated', or 'noop'")
    document_id: int = Field(..., description="Internal document ID")
    source: str = Field(..., description="Echo of the source")
    external_id: str = Field(..., description="Echo of the external_id")
    changed: bool = Field(..., description="Whether the document content/metadata was changed")
