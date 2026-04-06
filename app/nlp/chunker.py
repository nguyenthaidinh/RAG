"""
Semantic-first text chunker for the NLP pipeline.

Strategy:
  1. Split cleaned text into **paragraphs** (double-newline / heading boundaries).
  2. Greedily accumulate paragraphs into a chunk until ``max_tokens`` would
     be exceeded.
  3. If a single paragraph is larger than ``max_tokens``, delegate to the
     tokenizer's ``split()`` for sub-paragraph segmentation.
  4. Apply **overlap**: when finishing chunk *N*, carry the last
     ``overlap_tokens`` worth of trailing paragraphs into chunk *N+1*
     so that cross-boundary context is preserved.

Every output ``Chunk`` carries a SHA-256 ``content_hash`` and a
``token_count`` verified against the tokenizer.
"""
from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from app.nlp.types import Chunk

if TYPE_CHECKING:
    from app.nlp.types import Tokenizer

logger = logging.getLogger(__name__)

# Insert a paragraph break before markdown headings that are glued
# to the preceding paragraph (i.e. only one \n before #).
_RE_HEADING_BREAK = re.compile(r"(?<!\n)\n(#{1,6}\s)")
_RE_PARAGRAPH_SEP = re.compile(r"\n\s*\n")


class SemanticChunker:
    """
    Paragraph-aware chunker that respects semantic boundaries.

    Parameters
    ----------
    tokenizer : Tokenizer
        Any object satisfying the ``Tokenizer`` protocol.
    max_tokens : int
        Hard ceiling on tokens per chunk.
    overlap_tokens : int
        Number of trailing tokens from chunk *N* to repeat at the
        start of chunk *N+1*.
    """

    __slots__ = ("tokenizer", "max_tokens", "overlap_tokens")

    def __init__(
        self,
        tokenizer: Tokenizer,
        *,
        max_tokens: int = 512,
        overlap_tokens: int = 50,
    ) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if overlap_tokens < 0:
            raise ValueError("overlap_tokens must be >= 0")
        if overlap_tokens >= max_tokens:
            raise ValueError("overlap_tokens must be < max_tokens")
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    # ── public API ────────────────────────────────────────────────

    def chunk(
        self,
        text: str,
        *,
        tenant_id: str = "",
        document_id: int = 0,
        version_id: str = "",
    ) -> list[Chunk]:
        """
        Split *text* into a list of ``Chunk`` objects.

        Domain metadata (tenant_id, document_id, version_id) is attached
        to every chunk for downstream traceability.
        """
        paragraphs = self._split_paragraphs(text)
        if not paragraphs:
            return []

        chunks: list[Chunk] = []
        buf_parts: list[str] = []

        for para in paragraphs:
            para_tokens = self.tokenizer.count(para)

            # ── oversized paragraph: flush buffer, then sub-split ──
            if para_tokens > self.max_tokens:
                if buf_parts:
                    chunks.append(
                        self._build_chunk(buf_parts, len(chunks),
                                          tenant_id, document_id, version_id)
                    )
                    buf_parts = []

                sub_texts = self.tokenizer.split(
                    para, self.max_tokens, self.overlap_tokens,
                )
                for sub in sub_texts:
                    chunks.append(
                        self._build_chunk([sub], len(chunks),
                                          tenant_id, document_id, version_id)
                    )
                continue

            # ── would adding this paragraph exceed the budget? ──
            # Use the *actual* joined count (not a running sum) so that
            # ratio-based tokenizers cannot drift past max_tokens.
            if buf_parts and self._joined_count(buf_parts + [para]) > self.max_tokens:
                chunks.append(
                    self._build_chunk(buf_parts, len(chunks),
                                      tenant_id, document_id, version_id)
                )
                buf_parts = self._carry_overlap(buf_parts)

                # If overlap + new paragraph still exceeds, drop overlap
                if buf_parts and self._joined_count(buf_parts + [para]) > self.max_tokens:
                    buf_parts = []

            buf_parts.append(para)

        # ── flush remaining ──
        if buf_parts:
            chunks.append(
                self._build_chunk(buf_parts, len(chunks),
                                  tenant_id, document_id, version_id)
            )

        return chunks

    # ── internals ─────────────────────────────────────────────────

    @staticmethod
    def _split_paragraphs(text: str) -> list[str]:
        """Split on double-newline boundaries, with heading awareness."""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = _RE_HEADING_BREAK.sub(r"\n\n\1", text)
        parts = _RE_PARAGRAPH_SEP.split(text)
        return [p.strip() for p in parts if p.strip()]

    def _joined_count(self, parts: list[str]) -> int:
        """Token count of parts joined with paragraph separators."""
        return self.tokenizer.count("\n\n".join(parts))

    def _carry_overlap(self, parts: list[str]) -> list[str]:
        """
        Return tail paragraphs from *parts* that fit within
        ``self.overlap_tokens``.
        """
        if self.overlap_tokens <= 0:
            return []
        carry: list[str] = []
        carry_tokens = 0
        for p in reversed(parts):
            p_count = self.tokenizer.count(p)
            if carry_tokens + p_count > self.overlap_tokens:
                break
            carry.insert(0, p)
            carry_tokens += p_count
        return carry

    def _build_chunk(
        self,
        parts: list[str],
        index: int,
        tenant_id: str,
        document_id: int,
        version_id: str,
    ) -> Chunk:
        text = "\n\n".join(parts)
        token_count = self.tokenizer.count(text)
        return Chunk(
            chunk_index=index,
            text=text,
            token_count=token_count,
            content_hash=Chunk.compute_hash(text),
            tenant_id=tenant_id,
            document_id=document_id,
            version_id=version_id,
        )
