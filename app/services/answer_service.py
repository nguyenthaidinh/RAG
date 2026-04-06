from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnswerSnippet:
    document_id: int
    chunk_id: int
    snippet: str
    score: float | None = None
    source_document_id: int | None = None
    title: str | None = None
    heading: str | None = None
    debug_meta: dict | None = None


@dataclass(frozen=True)
class AnswerEvidence:
    """Evidence selected for answer generation.

    Represents a snippet chosen from the larger retrieval set as
    input context for the LLM. Citations map to evidences, not all results.
    """
    document_id: int
    chunk_id: int
    source_document_id: int | None
    score: float | None
    snippet: str
    title: str | None = None
    heading: str | None = None
    rank: int = 0


@dataclass(frozen=True)
class GeneratedAnswer:
    """Structured answer from the answer engine.

    Bundles generated text with the evidences actually used,
    intent classification, and provider metadata.
    Primary output of generate_structured().
    """
    text: str | None
    intent: str
    evidences: tuple[AnswerEvidence, ...]
    model: str | None = None
    provider: str | None = None
    used_history: bool = False


class AnswerService:
    """
    RAG answer synthesis (best-effort / fail-open).

    Phase 2B: supports three source modes:
      - knowledge-only: answer from document snippets (default, legacy)
      - system-only: answer from system context alone (no document snippets)
      - mixed: answer from both document snippets + system context

    Goals:
    - Never breaks retrieval response.
    - Avoid hallucination.
    - Produce more natural, focused answers from provided context.
    - Keep logs safe (no raw context / no raw prompt logging).
    """

    OVERVIEW_KEYWORDS = (
        "tóm tắt",
        "tổng quan",
        "nói về vấn đề gì",
        "nói về gì",
        "overview",
        "summary",
        "main idea",
        "main ideas",
        "key points",
    )

    SPECIFIC_KEYWORDS = (
        "điều nào",
        "khoản nào",
        "mục nào",
        "trách nhiệm",
        "quy trình",
        "các bước",
        "bước nào",
        "chi tiết",
        "cụ thể",
        "how",
        "which section",
        "responsibility",
        "procedure",
        "steps",
    )

    COMPARE_KEYWORDS = (
        "so sánh",
        "khác nhau",
        "giống nhau",
        "khác biệt",
        "tương đồng",
        "compare",
        "difference",
        "similarities",
        "versus",
        "vs",
    )

    def _detect_intent(self, question: str) -> str:
        q = (question or "").strip().lower()
        if any(k in q for k in self.OVERVIEW_KEYWORDS):
            return "overview"
        if any(k in q for k in self.COMPARE_KEYWORDS):
            return "compare"
        if any(k in q for k in self.SPECIFIC_KEYWORDS):
            return "specific"
        return "general"

    # ── System prompt ────────────────────────────────────────────────

    def _system_prompt(
        self,
        intent: str,
        *,
        has_history: bool = False,
        has_system_context: bool = False,
        has_document_context: bool = True,
    ) -> str:
        """Build the system prompt for LLM.

        Phase 2B: source precedence depends on what context is available.
        - Both doc + system → document is primary for facts, system for live state.
        - System-only → answer from system context only.
        - Document-only → answer from document context only (legacy).
        """
        # ── Evidence-grounding rules (shared across all modes) ────────
        grounding_rules = (
            "CRITICAL RULES FOR ANSWER QUALITY:\n"
            "- Use CONCRETE details from the context: exact names, states, rules, conditions, numbers, steps, fields, or services when the context provides them.\n"
            "- Do NOT paraphrase into vague or generic wording when the context has specific facts.\n"
            "- Answer the question DIRECTLY in your first sentence, then add supporting details.\n"
            "- If the context only partially answers the question, answer the supported part and say clearly what is not covered.\n"
            "- If multiple snippets repeat the same idea, merge into one coherent point — do not repeat.\n"
            "- Do NOT invent facts beyond what the context provides.\n"
            "- Do NOT use filler phrases like 'the system has appropriate mechanisms' or 'ensures safety and efficiency' — be specific about WHAT mechanism or HOW.\n"
            "- Use the same language as the user's question.\n"
        )

        if has_document_context and has_system_context:
            # MIXED mode: both sources available
            base = (
                "You are an internal assistant with access to document context and system context.\n"
                "Answer ONLY from the provided context.\n"
                "Document context is the PRIMARY source of truth for document-based and factual knowledge.\n"
                "System context is the PRIMARY source for current user, tenant, permission, and live system-state information.\n"
                "If both are present, prefer document context for document facts and system context for current user/tenant/access/live-state details.\n"
                + grounding_rules
            )
        elif has_system_context and not has_document_context:
            # SYSTEM-ONLY mode
            base = (
                "You are an internal assistant answering from system context.\n"
                "Answer ONLY from the provided system context.\n"
                "System context contains information about the current user, tenant, permissions, statistics, records, and workflows.\n"
                + grounding_rules
            )
        else:
            # Document-only mode (legacy Phase 1 behavior)
            base = (
                "You are an internal document question-answering assistant.\n"
                "Answer ONLY from the provided document context.\n"
                + grounding_rules
            )

        if has_history:
            base += (
                "\nIMPORTANT — Conversation history rules:\n"
                "- Use the recent conversation ONLY to resolve references (e.g. 'it', 'that', 'sau đó', 'cái đó') and understand what the current question is about.\n"
                "- The provided context is the PRIMARY source of truth for your answer.\n"
                "- If the conversation history conflicts with the provided context, ALWAYS follow the provided context.\n"
                "- Do NOT answer solely from conversation history when provided context is available.\n"
            )

        if intent == "overview":
            return (
                base
                + "\nFor this overview/summary question:\n"
                + "- Start with one direct summary sentence.\n"
                + "- Then list the specific main points, rules, or components FOUND in the context.\n"
                + "- Each point should contain a concrete detail, not a generic restatement.\n"
            )

        if intent == "compare":
            return (
                base
                + "\nFor this comparison question:\n"
                + "- Identify the exact items being compared.\n"
                + "- State specific differences and similarities FOUND in the context.\n"
                + "- Do NOT say 'there are differences' without listing them.\n"
            )

        if intent == "specific":
            return (
                base
                + "\nFor this specific/detail question:\n"
                + "- Answer with the exact detail requested — name, rule, step, condition, or value.\n"
                + "- If the context describes a workflow or process, state the steps/states in order.\n"
                + "- Prefer listing concrete items over writing prose.\n"
            )

        return (
            base
            + "\nGive a direct answer first using concrete details from the context, then add only the most relevant supporting facts.\n"
        )

    def _normalize_snippet(self, text: str) -> str:
        text = " ".join((text or "").split()).strip().lower()
        return text

    def _build_context(self, snippets: list[AnswerSnippet]) -> str:
        max_results = int(getattr(settings, "LLM_ANSWER_MAX_RESULTS", 5))
        max_snippet_chars = int(getattr(settings, "LLM_ANSWER_MAX_SNIPPET_CHARS", 1000))
        max_context_chars = int(getattr(settings, "LLM_ANSWER_MAX_CONTEXT_CHARS", 10000))

        parts: list[str] = []
        seen: set[str] = set()

        for i, s in enumerate(snippets, start=1):
            text = (s.snippet or "").strip()
            if not text:
                continue

            # truncate each snippet early
            text = text[:max_snippet_chars].strip()

            # dedupe near-identical snippets
            norm = self._normalize_snippet(text[:400])
            if norm in seen:
                continue
            seen.add(norm)

            parts.append(f"[{len(parts)+1}] doc={s.document_id} chunk={s.chunk_id}\n{text}")

            if len(parts) >= max_results:
                break

        context = "\n\n".join(parts).strip()
        if len(context) > max_context_chars:
            context = context[:max_context_chars].strip()

        return context

    # ── History helpers ──────────────────────────────────────────────

    _MAX_HISTORY_TURNS = 8
    _MAX_TURN_CHARS = 300
    _MAX_HISTORY_BLOCK_CHARS = 1500

    def _build_history_block(self, history: list[Any]) -> str:
        """
        Build a compact conversation history block for prompt injection.

        Rules:
        - Take at most _MAX_HISTORY_TURNS most recent turns
        - Trim each turn's text to _MAX_TURN_CHARS
        - Skip empty turns
        - Cap total block at _MAX_HISTORY_BLOCK_CHARS
        """
        if not history:
            return ""

        recent = history[-self._MAX_HISTORY_TURNS:]
        lines: list[str] = []

        for turn in recent:
            role = getattr(turn, "role", None) or (turn.get("role") if isinstance(turn, dict) else None)
            text = getattr(turn, "text", None) or (turn.get("text") if isinstance(turn, dict) else None)

            if not role or not text or not text.strip():
                continue

            text = text.strip()
            if len(text) > self._MAX_TURN_CHARS:
                text = text[:self._MAX_TURN_CHARS] + "..."

            label = "User" if role == "user" else "Assistant"
            lines.append(f"{label}: {text}")

        if not lines:
            return ""

        block = "\n".join(lines)
        if len(block) > self._MAX_HISTORY_BLOCK_CHARS:
            block = block[:self._MAX_HISTORY_BLOCK_CHARS] + "..."

        return block

    # ── User prompt ──────────────────────────────────────────────────

    def _user_prompt(
        self,
        question: str,
        context: str,
        intent: str,
        *,
        history_block: str = "",
        system_context_block: str = "",
    ) -> str:
        """Build user prompt with conditional sections.

        Phase 2B: empty document context and empty system context blocks
        are excluded from the prompt entirely — no empty sections rendered.
        """
        instruction = (
            "Write a direct, evidence-grounded answer for the user.\n"
            "Use specific details (names, rules, steps, conditions, values) from the context — do not paraphrase into vague statements.\n"
            "Do not repeat duplicated information.\n"
            "Do not mention chunk numbers unless necessary.\n"
        )

        if intent == "overview":
            instruction += (
                "Give one direct summary sentence, then list the concrete main points found in the context.\n"
            )
        elif intent == "compare":
            instruction += (
                "State specific differences and similarities found in the context.\n"
                "Do not just say 'there are differences' — list them.\n"
            )
        elif intent == "specific":
            instruction += (
                "Answer with the exact detail requested.\n"
                "If the answer contains steps, rules, or conditions, list them concretely.\n"
            )
        else:
            instruction += "Answer directly using concrete facts from the context.\n"

        # Build prompt parts in order: question → history → system context → document context
        parts = [f"User question:\n{question.strip()}"]

        if history_block:
            parts.append(f"Recent conversation:\n{history_block}")

        if system_context_block:
            parts.append(f"System context:\n{system_context_block}")

        if context.strip():
            parts.append(f"Document context:\n{context.strip()}")

        parts.append(f"{instruction}\nAnswer:")

        return "\n\n".join(parts)

    # ── Evidence selection layer ─────────────────────────────────────

    _MAX_EVIDENCES: int = 4

    def _dedupe_snippets(
        self, snippets: list[AnswerSnippet],
    ) -> list[AnswerSnippet]:
        """Remove near-duplicate snippets based on normalized text prefix."""
        seen: set[str] = set()
        deduped: list[AnswerSnippet] = []
        for s in snippets:
            norm = self._normalize_snippet((s.snippet or "")[:400])
            if not norm or norm in seen:
                continue
            seen.add(norm)
            deduped.append(s)
        return deduped

    def _select_evidences(
        self,
        snippets: list[AnswerSnippet],
        question: str,
        intent: str,
    ) -> list[AnswerEvidence]:
        """Select top evidences for answer generation.

        Strategy (Phase 1 — deliberately simple):
          1. Dedupe by normalized text similarity
          2. Sort by score descending
          3. Take top-N (configurable via _MAX_EVIDENCES)
          4. Assign rank (1 = most relevant)
        """
        if not snippets:
            return []

        valid = [s for s in snippets if s.snippet and s.snippet.strip()]
        if not valid:
            return []

        deduped = self._dedupe_snippets(valid)
        scored = sorted(deduped, key=lambda s: s.score or 0.0, reverse=True)
        selected = scored[:self._MAX_EVIDENCES]

        evidences: list[AnswerEvidence] = []
        for rank_idx, s in enumerate(selected, start=1):
            evidences.append(
                AnswerEvidence(
                    document_id=s.document_id,
                    chunk_id=s.chunk_id,
                    source_document_id=s.source_document_id,
                    score=s.score,
                    snippet=s.snippet,
                    title=s.title,
                    heading=s.heading,
                    rank=rank_idx,
                )
            )

        return evidences

    def _build_context_from_evidences(
        self, evidences: list[AnswerEvidence],
    ) -> str:
        """Build LLM context from selected evidences.

        Similar to _build_context() but operates on the already-selected,
        already-deduped evidence list.
        """
        max_snippet_chars = int(
            getattr(settings, "LLM_ANSWER_MAX_SNIPPET_CHARS", 1000)
        )
        max_context_chars = int(
            getattr(settings, "LLM_ANSWER_MAX_CONTEXT_CHARS", 10000)
        )

        parts: list[str] = []
        for e in evidences:
            text = (e.snippet or "").strip()
            if not text:
                continue
            text = text[:max_snippet_chars].strip()

            # Include title/heading for better context anchoring
            header = f"[{len(parts)+1}] doc={e.document_id} chunk={e.chunk_id}"
            if e.title:
                header += f" title=\"{e.title}\""
            if e.heading:
                header += f" section=\"{e.heading}\""

            parts.append(f"{header}\n{text}")

        context = "\n\n".join(parts).strip()
        if len(context) > max_context_chars:
            context = context[:max_context_chars].strip()

        return context

    async def _openai_chat(
        self,
        *,
        question: str,
        context: str,
        intent: str,
        history_block: str = "",
        system_context_block: str = "",
        has_document_context: bool = True,
    ) -> str | None:
        api_key = getattr(settings, "OPENAI_API_KEY", None)
        if not api_key:
            return None

        model = getattr(settings, "LLM_ANSWER_MODEL", "gpt-4o-mini")
        temperature = float(getattr(settings, "LLM_ANSWER_TEMPERATURE", 0.2))
        max_tokens = int(getattr(settings, "LLM_ANSWER_MAX_TOKENS", 600))
        timeout_s = float(getattr(settings, "LLM_ANSWER_TIMEOUT_S", 12.0))

        has_history = bool(history_block)
        has_sys_ctx = bool(system_context_block)

        payload = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {
                    "role": "system",
                    "content": self._system_prompt(
                        intent,
                        has_history=has_history,
                        has_system_context=has_sys_ctx,
                        has_document_context=has_document_context,
                    ),
                },
                {
                    "role": "user",
                    "content": self._user_prompt(
                        question, context, intent,
                        history_block=history_block,
                        system_context_block=system_context_block,
                    ),
                },
            ],
        }

        async with httpx.AsyncClient(timeout=timeout_s + 1.0) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        try:
            text = data["choices"][0]["message"]["content"]
        except Exception:
            return None

        if not isinstance(text, str):
            return None

        text = text.strip()
        return text if text else None

    # ── Main generation method ───────────────────────────────────────

    async def generate_structured(
        self,
        *,
        question: str,
        snippets: list[AnswerSnippet],
        history: list[Any] | None = None,
        system_context_block: str = "",
        question_category: str = "knowledge",
        allow_system_context_only: bool = False,
    ) -> GeneratedAnswer:
        """Generate a structured answer with evidence tracking.

        Phase 2B additions:
          - question_category: routing category from orchestrator
          - allow_system_context_only: if True and no evidences, still
            attempt LLM answer from system_context_block alone

        Evidence selection happens BEFORE the LLM call so callers
        always receive evidences even if the LLM fails or times out.

        Returns:
            GeneratedAnswer with text (may be None on LLM failure),
            selected evidences, intent, and provider metadata.
        """
        intent = self._detect_intent(question)

        # ── Evidence selection (BEFORE LLM) ──────────────────────────
        evidences = self._select_evidences(snippets, question, intent)
        has_document_evidence = bool(evidences)
        has_system_context = bool(system_context_block)

        # ── Gate: no content at all → early return ───────────────────
        if not has_document_evidence and not (allow_system_context_only and has_system_context):
            return GeneratedAnswer(text=None, intent=intent, evidences=())

        # ── Gate: LLM disabled → return evidences only ───────────────
        if not getattr(settings, "LLM_ANSWER_ENABLED", False):
            return GeneratedAnswer(
                text=None, intent=intent, evidences=tuple(evidences),
            )

        provider = (
            getattr(settings, "LLM_ANSWER_PROVIDER", "none") or "none"
        ).lower().strip()
        if provider != "openai":
            return GeneratedAnswer(
                text=None, intent=intent, evidences=tuple(evidences),
            )

        model = getattr(settings, "LLM_ANSWER_MODEL", "gpt-4o-mini")

        # ── Build context from selected evidences ────────────────────
        context = self._build_context_from_evidences(evidences)

        # Phase 2B: allow system-only path when no document context
        if not context and not (allow_system_context_only and has_system_context):
            return GeneratedAnswer(
                text=None, intent=intent, evidences=tuple(evidences),
            )

        # ── History block ────────────────────────────────────────────
        history_block = self._build_history_block(history or [])
        used_history = bool(history_block)
        history_turn_count = len(history) if history else 0

        # ── LLM call (fail-open) ─────────────────────────────────────
        timeout_s = float(getattr(settings, "LLM_ANSWER_TIMEOUT_S", 12.0))
        started_at = time.perf_counter()
        text: str | None = None

        try:
            text = await asyncio.wait_for(
                self._openai_chat(
                    question=question,
                    context=context,
                    intent=intent,
                    history_block=history_block,
                    system_context_block=system_context_block,
                    has_document_context=has_document_evidence,
                ),
                timeout=timeout_s,
            )
            elapsed_ms = int((time.perf_counter() - started_at) * 1000)

            logger.info(
                "answer.generate_ok provider=%s model=%s intent=%s "
                "question_category=%s snippets=%d evidences=%d "
                "context_chars=%d history_turns=%d used_history=%s "
                "used_system_context=%s sys_ctx_chars=%d "
                "has_document_evidence=%s latency_ms=%d",
                provider,
                model,
                intent,
                question_category,
                len(snippets),
                len(evidences),
                len(context),
                history_turn_count,
                used_history,
                has_system_context,
                len(system_context_block),
                has_document_evidence,
                elapsed_ms,
            )

        except asyncio.TimeoutError:
            logger.warning(
                "answer.generate_timeout provider=%s intent=%s "
                "question_category=%s timeout_s=%s",
                provider,
                intent,
                question_category,
                timeout_s,
            )

        except Exception as exc:
            logger.warning(
                "answer.generate_failed provider=%s intent=%s "
                "question_category=%s error=%s",
                provider,
                intent,
                question_category,
                exc.__class__.__name__,
            )

        return GeneratedAnswer(
            text=text,
            intent=intent,
            evidences=tuple(evidences),
            model=model if text is not None else None,
            provider=provider if text is not None else None,
            used_history=used_history,
        )

    async def generate(
        self,
        *,
        question: str,
        snippets: list[AnswerSnippet],
        history: list[Any] | None = None,
        system_context_block: str = "",
    ) -> str | None:
        """Backward-compatible wrapper — returns answer text only.

        Existing callers (e.g. /query endpoint) use this method.
        Delegates to generate_structured() internally.
        """
        result = await self.generate_structured(
            question=question,
            snippets=snippets,
            history=history,
            system_context_block=system_context_block,
        )
        return result.text