"""
Core types for the NLP processing pipeline.

Chunk is the unit of output from the chunker — carries text plus
metadata so downstream consumers (embeddings, search, etc.) can
operate without additional lookups.

Tokenizer is the vendor-agnostic protocol every tokenizer adapter
must satisfy.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(slots=True)
class Chunk:
    """A single text chunk produced by the chunker."""

    chunk_index: int
    text: str
    token_count: int
    content_hash: str

    # Domain metadata — filled by the pipeline, not the chunker itself.
    tenant_id: str = ""
    document_id: int = 0
    version_id: str = ""

    @staticmethod
    def compute_hash(text: str) -> str:
        """Deterministic SHA-256 of chunk text."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


@runtime_checkable
class Tokenizer(Protocol):
    """
    Vendor-agnostic tokenizer interface.

    Every provider (OpenAI / Gemini / local) must implement both methods.
    ``count`` and ``split`` MUST be mutually consistent: for every segment
    returned by ``split``, ``count(segment) <= max_tokens`` must hold.
    """

    def count(self, text: str) -> int:
        """Return the number of tokens in *text*."""
        ...

    def split(
        self,
        text: str,
        max_tokens: int,
        overlap_tokens: int = 0,
    ) -> list[str]:
        """
        Split *text* into segments each containing at most *max_tokens* tokens.

        Consecutive segments overlap by *overlap_tokens* tokens so that
        context is not lost at boundaries.
        """
        ...
