from pydantic import BaseModel, Field, field_validator
from typing import Any
from datetime import datetime
from app.schemas.ingest_mode import IngestMode


# class UpsertDocumentRequest(BaseModel):
#     source: str = Field(..., min_length=1, max_length=50, description="Source identifier (e.g. web, pdf, api, manual)")
#     external_id: str = Field(..., min_length=1, max_length=512, description="External identifier unique per tenant+source")
#     title: str | None = Field(None, max_length=512, description="Document title")
#     content: str = Field(..., min_length=1, description="Raw content (HTML or plain text)")
#     metadata: dict[str, Any] = Field(default_factory=dict, description="Optional metadata JSON object")
    
#     @field_validator("content")
#     @classmethod
#     def validate_content_size(cls, v: str) -> str:
#         # Limit content to 5MB (5 * 1024 * 1024 bytes)
#         max_size = 5 * 1024 * 1024
#         if len(v.encode("utf-8")) > max_size:
#             raise ValueError(f"Content size exceeds maximum of {max_size} bytes (5MB)")
#         return v
    

class UpsertDocumentRequest(BaseModel):
    source: str = Field(..., min_length=1, max_length=50, description="Source identifier (e.g. web, pdf, api, manual)")
    external_id: str = Field(..., min_length=1, max_length=512, description="External identifier unique per tenant+source")
    title: str | None = Field(None, max_length=512, description="Document title")
    content: str = Field(..., min_length=1, description="Raw content (HTML or plain text)")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Optional metadata JSON object")
    ingest_mode: IngestMode | None = Field(
        None,
        description="Ingest mode: legacy | semantic"
    )


class UpsertDocumentResponse(BaseModel):
    status: str = Field(..., description="Response status")
    action: str = Field(..., description="Action taken: created, updated, or noop")
    document_id: int = Field(..., description="Document ID")
    changed: bool = Field(..., description="Whether the document was changed")


class DocumentResponse(BaseModel):
    model_config = {"from_attributes": True, "populate_by_name": True}

    id: int
    tenant_id: str
    source: str
    external_id: str
    title: str | None
    status: str
    metadata: dict[str, Any] = Field(alias="meta")
    created_at: datetime
    updated_at: datetime
