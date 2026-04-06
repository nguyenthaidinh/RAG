"""
Document Synthesis Engine (Phase 9.0 — Step 2).

Standalone AI synthesis service that transforms raw document text into
a cleaner, structured, retrieval-friendly markdown knowledge document.

This module is the "engine" only — it does NOT persist anything to DB,
does NOT create child documents, does NOT trigger any downstream pipeline.
Step 3 will later wire ingest orchestration to call this service.

Provider: OpenAI (via httpx, same pattern as AnswerService).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────

PROMPT_VERSION = "v1"

# Safe ceiling to avoid enormous prompt payloads.
# ~100k chars ≈ ~25k tokens — leaves room for system prompt + output.
MAX_INPUT_CHARS = 100_000

OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"


# ── Exceptions ───────────────────────────────────────────────────────────

class DocumentSynthesisError(Exception):
    """Raised when document synthesis fails."""


# ── Result dataclass ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class SynthesizedDocumentResult:
    """Immutable result of a document synthesis operation."""

    content_markdown: str
    summary: str
    provider: str
    model: str
    prompt_version: str
    input_chars: int
    output_chars: int


# ── System prompt ────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a document restructuring assistant.

Your task: transform the user-provided source text into a clearer, \
well-structured markdown knowledge document optimized for search and retrieval.

STRICT RULES:
1. Stay FAITHFUL to the source content. Do NOT invent facts.
2. Do NOT add outside knowledge beyond what the source provides.
3. Do NOT silently guess or fill in missing details.
4. If the source is unclear, incomplete, or ambiguous, explicitly state that \
   rather than hallucinating.
5. Write all body content (paragraphs, bullet text, descriptions) in the \
   SAME language as the source document.
6. Section headings MUST be exactly these English headings — do NOT \
   translate them:
   - `## Summary`
   - `## Key Points`
   - `## Main Content`
   - `## Important Notes`
   Sub-section headings under "## Main Content" (e.g. `### <topic>`) \
   may use the source language.

OUTPUT FORMAT (markdown):
# <Title or synthesized title — same language as source>

## Summary
<1-3 sentence summary, in the source language>

## Key Points
- <point 1>
- <point 2>
- ...

## Main Content
### <Section 1 heading>
<restructured content>

### <Section 2 heading>
<restructured content>

(add more sections as needed)

## Important Notes
- <any caveats, limitations, or unclear areas from the source>

Keep the output concise, well-organized, and easy to chunk for retrieval.\
"""


# ── Service ──────────────────────────────────────────────────────────────

class DocumentSynthesisService:
    """
    Standalone AI document synthesis engine.

    Transforms raw document text into a structured, retrieval-friendly
    markdown knowledge document using OpenAI.

    Usage::

        svc = DocumentSynthesisService()
        result = await svc.synthesize_document(
            title="My Document",
            content_text="...",
            document_id=42,
            tenant_id="acme",
        )
        # result.content_markdown contains the synthesized markdown
    """

    async def synthesize_document(
        self,
        *,
        title: str | None,
        content_text: str,
        document_id: int | None = None,
        tenant_id: str | None = None,
    ) -> SynthesizedDocumentResult:
        """
        Synthesize a structured knowledge document from raw text.

        Args:
            title: Optional document title (used as hint in prompt).
            content_text: Raw document text to transform.
            document_id: Optional, for logging context only.
            tenant_id: Optional, for logging context only.

        Returns:
            SynthesizedDocumentResult with markdown content and metadata.

        Raises:
            DocumentSynthesisError: If validation fails or OpenAI call fails.
        """
        # ── Kill switch ───────────────────────────────────────────────
        if not settings.SYNTHESIS_ENABLED:
            raise DocumentSynthesisError("Document synthesis is disabled")

        # ── Validate input ───────────────────────────────────────────
        content_text = self._validate_and_prepare(content_text)
        input_chars = len(content_text)

        # ── Resolve provider settings ────────────────────────────────
        model = settings.SYNTHESIS_MODEL
        provider = "openai"

        logger.info(
            "synthesis.start tenant_id=%s doc_id=%s input_chars=%d "
            "provider=%s model=%s prompt_version=%s",
            tenant_id, document_id, input_chars,
            provider, model, PROMPT_VERSION,
        )

        # ── Call OpenAI ──────────────────────────────────────────────
        raw_output = await self._openai_synthesize(
            title=title,
            content_text=content_text,
            model=model,
            document_id=document_id,
            tenant_id=tenant_id,
        )

        # ── Extract summary from output ─────────────────────────────
        summary = self._extract_summary(raw_output)

        result = SynthesizedDocumentResult(
            content_markdown=raw_output,
            summary=summary,
            provider=provider,
            model=model,
            prompt_version=PROMPT_VERSION,
            input_chars=input_chars,
            output_chars=len(raw_output),
        )

        logger.info(
            "synthesis.done tenant_id=%s doc_id=%s input_chars=%d "
            "output_chars=%d provider=%s model=%s",
            tenant_id, document_id, result.input_chars,
            result.output_chars, provider, model,
        )

        return result

    # ── Internal helpers ─────────────────────────────────────────────

    @staticmethod
    def _validate_and_prepare(content_text: str) -> str:
        """Validate and safely prepare input text."""
        if not content_text or not content_text.strip():
            raise DocumentSynthesisError("Content text is empty or blank")

        content_text = content_text.strip()

        # Truncate safely if too long (cut at char boundary, not mid-word)
        if len(content_text) > MAX_INPUT_CHARS:
            content_text = content_text[:MAX_INPUT_CHARS]
            # Find last space to avoid cutting mid-word
            last_space = content_text.rfind(" ", MAX_INPUT_CHARS - 200)
            if last_space > 0:
                content_text = content_text[:last_space]

        return content_text

    def _build_user_prompt(self, title: str | None, content_text: str) -> str:
        """Build the user message for OpenAI."""
        parts: list[str] = []
        if title and title.strip():
            parts.append(f"Document title: {title.strip()}")
        parts.append(f"Source content:\n{content_text}")
        return "\n\n".join(parts)

    async def _openai_synthesize(
        self,
        *,
        title: str | None,
        content_text: str,
        model: str,
        document_id: int | None,
        tenant_id: str | None,
    ) -> str:
        """Call OpenAI Chat API to synthesize document."""
        api_key = settings.OPENAI_API_KEY
        if not api_key:
            raise DocumentSynthesisError(
                "OPENAI_API_KEY not configured"
            )

        timeout_s = settings.SYNTHESIS_TIMEOUT_S
        max_tokens = settings.SYNTHESIS_MAX_TOKENS
        temperature = settings.SYNTHESIS_TEMPERATURE

        payload = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": self._build_user_prompt(title, content_text)},
            ],
        }

        try:
            async with httpx.AsyncClient(timeout=timeout_s + 1.0) as client:
                resp = await client.post(
                    OPENAI_CHAT_URL,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as exc:
            # Log status only, never raw prompt/response body
            logger.error(
                "synthesis.openai_error tenant_id=%s doc_id=%s status=%d",
                tenant_id, document_id, exc.response.status_code,
            )
            raise DocumentSynthesisError(
                f"OpenAI request failed with status {exc.response.status_code}"
            ) from exc
        except httpx.TimeoutException as exc:
            logger.error(
                "synthesis.openai_timeout tenant_id=%s doc_id=%s timeout_s=%.1f",
                tenant_id, document_id, timeout_s,
            )
            raise DocumentSynthesisError(
                f"OpenAI request timed out after {timeout_s}s"
            ) from exc
        except Exception as exc:
            logger.error(
                "synthesis.openai_failed tenant_id=%s doc_id=%s error_type=%s",
                tenant_id, document_id, type(exc).__name__,
            )
            raise DocumentSynthesisError(
                "OpenAI request failed unexpectedly"
            ) from exc

        # ── Parse response ───────────────────────────────────────────
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise DocumentSynthesisError(
                "Invalid response structure from OpenAI"
            ) from exc

        if not isinstance(text, str) or not text.strip():
            raise DocumentSynthesisError(
                "Empty response from OpenAI"
            )

        return text.strip()

    @staticmethod
    def _extract_summary(markdown: str) -> str:
        """
        Extract summary section from synthesized markdown.

        Looks for content between '## Summary' and the next '##' heading.
        Falls back to first 200 chars if summary section not found.
        """
        lines = markdown.split("\n")
        in_summary = False
        summary_lines: list[str] = []

        for line in lines:
            stripped = line.strip()
            if stripped == "## Summary":
                in_summary = True
                continue
            if in_summary:
                if stripped.startswith("## "):
                    break
                if stripped:
                    summary_lines.append(stripped)

        if summary_lines:
            return " ".join(summary_lines)

        # Fallback: first meaningful line, truncated
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                return stripped[:200]

        return markdown[:200]
