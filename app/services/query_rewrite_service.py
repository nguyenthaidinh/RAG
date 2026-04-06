"""
Query Rewrite Service V2 (Phase 3A).

Guarded query preprocessing layer that sits BEFORE retrieval.
Produces a RetrievalPlan with:
  - original_query (always preserved)
  - optional rewritten_query
  - optional step_back_query
  - optional subqueries (max 2)
  - query_mode classification
  - rewrite_strategy gating decision
  - confidence score
  - telemetry

V2 improvements over V1:
  - Behavior-class classification (QueryMode = query shape)
  - Constraint detection as orthogonal signal (not a query mode)
  - Gating via RewriteStrategy (no_rewrite / light_normalize /
    contextual_rewrite / controlled_decomposition / safe_fallback)
  - Basic constraint preservation guardrail: year/date numbers and
    negation keywords are checked after LLM rewrite. Other constraint
    types (role, unit, contract) are detected for gating decisions
    but NOT yet verified in post-rewrite guardrails.
  - Disciplined history usage: only recent turns, only when follow-up
    detected, never "guess" from history when confidence is low
  - Upgraded LLM prompt with constraint-preservation instructions
  - Better telemetry (rewrite_strategy logged, no raw text)

Design rules:
  - Feature-flagged via QUERY_REWRITE_ENABLED
  - Fail-open: any error → passthrough original query
  - No hallucinated filters/metadata
  - No invented facts — rewrite only clarifies intent
  - Max 1 rewritten + 1 step_back + 2 subqueries
  - Timeout-protected LLM call
  - No raw content in logs
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any

import httpx

from app.core.config import settings
from app.schemas.query_rewrite import QueryMode, RetrievalPlan, RewriteStrategy

logger = logging.getLogger(__name__)

# ── Follow-up markers ────────────────────────────────────────────────

_FOLLOW_UP_MARKERS_VI = (
    "cái này", "cái đó", "cái kia",
    "trường hợp đó", "trường hợp này",
    "còn cái", "vậy thì", "thế thì",
    "ngoài ra", "về vấn đề",
    "ở trên", "nói ở trên",
    "vừa nói", "bạn vừa nói",
    "như trên", "phần trước",
    "tiếp tục", "tiếp theo",
)

_FOLLOW_UP_MARKERS_EN = (
    "this one", "that one",
    "the one you", "you mentioned",
    "mentioned earlier", "said above",
    "as above", "the previous",
    "what about", "how about",
    "and also", "additionally",
    "on top of", "regarding that",
    "in that case", "following up",
)

# ── Intent keywords (reuse from AnswerService with extensions) ────────

_OVERVIEW_KW = (
    "tóm tắt", "tổng quan", "nói về gì", "overview", "summary",
    "main idea", "key points", "giới thiệu",
)
_SPECIFIC_KW = (
    "điều nào", "khoản nào", "mục nào", "trách nhiệm", "quy trình",
    "các bước", "chi tiết", "cụ thể", "how", "which section",
    "responsibility", "procedure", "steps", "quy định",
)
_COMPARE_KW = (
    "so sánh", "khác nhau", "giống nhau", "khác biệt",
    "compare", "difference", "similarities", "versus", "vs",
)
_MULTI_HOP_KW = (
    "liên quan đến", "ảnh hưởng đến", "does it affect",
    "relationship between", "cả hai", "both", "nếu mà",
    "trong trường hợp nào",
)

# ── V2: Constraint detection patterns ────────────────────────────────

_CONSTRAINT_PATTERNS = (
    # Year / time period
    re.compile(r"\b(năm\s+)?20\d{2}\b", re.IGNORECASE),
    re.compile(r"\b(tháng|quý|kỳ|học kỳ|semester|quarter)\s+\d", re.IGNORECASE),
    # Negation (Vietnamese + English)
    re.compile(r"\b(không|chưa|không được|không phải|chẳng|không bắt buộc)\b", re.IGNORECASE),
    re.compile(r"\b(not|never|no longer|without|cannot|must not)\b", re.IGNORECASE),
    # Unit / department / level
    re.compile(r"\b(khoa|phòng|ban|bộ phận|cấp|department|unit|division)\b", re.IGNORECASE),
    # Role / subject
    re.compile(r"\b(giảng viên|sinh viên|nhân viên|cán bộ|lecturer|student|staff)\b", re.IGNORECASE),
    # Contract type
    re.compile(r"\b(hợp đồng|biên chế|thử việc|contract|permanent|probation)\b", re.IGNORECASE),
    # Permission / condition keywords
    re.compile(r"\b(ai được|ai phải|điều kiện|bắt buộc|phải|được phép)\b", re.IGNORECASE),
    re.compile(r"\b(eligible|required|mandatory|allowed|permission)\b", re.IGNORECASE),
)

_MIN_QUERY_LENGTH_FOR_REWRITE = 4  # very short queries are likely direct
_MAX_HISTORY_TURNS = 4  # max turns to use from history
_MAX_HISTORY_TEXT_LEN = 200  # max chars per turn

# ── Forbidden filter tokens ──────────────────────────────────────────
# LLM sometimes hallucinates filter-like prefixes that our retrieval
# engine does NOT support.  Reject queries containing these.

_FORBIDDEN_FILTER_RE = re.compile(
    r"\b(?:tenant_id|document_id|doc_id|source|metadata|tag|filter)\s*:",
    re.IGNORECASE,
)

_MIN_VALID_QUERY_LENGTH = 2  # minimum chars for a valid query


class QueryRewriteService:
    """
    Guarded query rewrite engine (V2).

    Public API:
      - maybe_rewrite(query_text, history) → RetrievalPlan

    V2 additions:
      - Constraint-aware classification
      - RewriteStrategy gating
      - Better history resolution
      - Constraint-preserving guardrails
    """

    def __init__(self) -> None:
        self._enabled = getattr(settings, "QUERY_REWRITE_ENABLED", False)
        self._provider = (
            getattr(settings, "QUERY_REWRITE_PROVIDER", "none") or "none"
        ).lower().strip()
        self._model = getattr(settings, "QUERY_REWRITE_MODEL", "gpt-4o-mini")
        self._timeout = float(getattr(settings, "QUERY_REWRITE_TIMEOUT_S", 3.0))
        self._max_tokens = int(getattr(settings, "QUERY_REWRITE_MAX_TOKENS", 300))
        self._temperature = float(getattr(settings, "QUERY_REWRITE_TEMPERATURE", 0.1))
        self._max_subqueries = int(getattr(settings, "QUERY_REWRITE_MAX_SUBQUERIES", 2))
        self._max_query_chars = int(getattr(settings, "QUERY_REWRITE_MAX_QUERY_CHARS", 1200))
        self._confidence_threshold = float(
            getattr(settings, "QUERY_REWRITE_CONFIDENCE_THRESHOLD", 0.5)
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ── Validation helpers ───────────────────────────────────────────

    @staticmethod
    def _is_valid_query(q: str | None) -> bool:
        """Check if a query is valid for retrieval.

        Rejects:
          - None / empty / whitespace-only
          - Too short (< 2 chars of actual content)
          - Purely punctuation / symbols
          - Contains hallucinated filter tokens
        """
        if not q:
            return False
        q = q.strip()
        if len(q) < _MIN_VALID_QUERY_LENGTH:
            return False
        # reject purely punctuation / whitespace
        cleaned = re.sub(r"[^\w]", "", q)
        if not cleaned:
            return False
        # reject queries with forbidden filter tokens
        if _FORBIDDEN_FILTER_RE.search(q):
            return False
        return True

    @staticmethod
    def _has_forbidden_filters(q: str) -> bool:
        """Check if query contains hallucinated filter tokens."""
        return bool(_FORBIDDEN_FILTER_RE.search(q))

    # ── Main entry ───────────────────────────────────────────────────

    async def maybe_rewrite(
        self,
        query_text: str,
        history: list[Any] | None = None,
    ) -> RetrievalPlan:
        """
        Analyze and optionally rewrite a query for better retrieval.

        V2 flow:
          1. Classify query mode (behavior class)
          2. Detect constraints
          3. Determine rewrite strategy (gating)
          4. Execute strategy
          5. Apply guardrails
          6. Telemetry

        Fail-open: any error → passthrough with original query.
        """
        t0 = time.perf_counter()
        query_text = (query_text or "").strip()

        if not query_text:
            return RetrievalPlan.passthrough("")

        # Gate: feature flag
        if not self._enabled:
            return RetrievalPlan.passthrough(query_text)

        try:
            # 1) Classify query mode
            query_mode = self._classify_query_mode(query_text, history)

            # 2) Detect constraints in query
            constraints = self._detect_constraints(query_text)
            has_constraints = len(constraints) > 0

            # 3) Determine rewrite strategy (gating)
            strategy = self._determine_strategy(
                query_text, query_mode, history, has_constraints,
            )

            # 4) Execute strategy
            rewritten = None
            subqueries: tuple[str, ...] = ()
            step_back = None
            confidence = 0.8
            reason = f"mode={query_mode.value},strategy={strategy.value}"
            used_history = False

            if strategy == RewriteStrategy.NO_REWRITE:
                # Query is clear — no rewrite needed
                confidence = 1.0
                reason = "clear_query_no_rewrite"

            elif strategy == RewriteStrategy.LIGHT_NORMALIZE:
                # Conservative pass — no LLM, no decomposition
                confidence = 0.95
                reason = "conservative_pass"

            elif strategy == RewriteStrategy.SAFE_FALLBACK:
                # Not enough confidence to rewrite safely
                confidence = 1.0
                reason = "safe_fallback_insufficient_context"

            elif strategy == RewriteStrategy.CONTEXTUAL_REWRITE:
                # Resolve follow-up references via history first
                resolved_query = query_text
                if history:
                    resolved = self._resolve_history_references(
                        query_text, history,
                    )
                    if resolved and resolved != query_text:
                        resolved_query = resolved
                        used_history = True

                # LLM rewrite if available
                if self._provider == "openai" and getattr(settings, "OPENAI_API_KEY", ""):
                    llm_result = await self._llm_rewrite(
                        resolved_query, query_mode, history,
                        constraints=constraints,
                    )
                    if llm_result:
                        rewritten = llm_result.get("rewritten_query")
                        step_back = llm_result.get("step_back_query")
                        raw_subs = llm_result.get("subqueries") or []
                        subqueries = tuple(raw_subs[:self._max_subqueries])
                        confidence = float(llm_result.get("confidence", 0.7))
                        reason = llm_result.get("reason", reason)

                # History resolution fallback
                if used_history and resolved_query != query_text and not rewritten:
                    rewritten = resolved_query
                    confidence = max(confidence, 0.7)
                    reason = "history_resolved_without_llm"

            elif strategy == RewriteStrategy.CONTROLLED_DECOMPOSITION:
                # LLM rewrite with decomposition intent
                if self._provider == "openai" and getattr(settings, "OPENAI_API_KEY", ""):
                    llm_result = await self._llm_rewrite(
                        query_text, query_mode, history,
                        constraints=constraints,
                    )
                    if llm_result:
                        rewritten = llm_result.get("rewritten_query")
                        step_back = llm_result.get("step_back_query")
                        raw_subs = llm_result.get("subqueries") or []
                        subqueries = tuple(raw_subs[:self._max_subqueries])
                        confidence = float(llm_result.get("confidence", 0.7))
                        reason = llm_result.get("reason", reason)

            # 5) Apply guardrails (constraint-preserving)
            plan = self._apply_guardrails(
                original_query=query_text,
                rewritten_query=rewritten,
                step_back_query=step_back,
                subqueries=subqueries,
                query_mode=query_mode,
                rewrite_strategy=strategy,
                confidence=confidence,
                reason=reason,
                used_history=used_history,
                constraints=constraints,
                latency_ms=int((time.perf_counter() - t0) * 1000),
            )

            # 6) Telemetry (no raw text)
            logger.info(
                "query_rewrite.done mode=%s strategy=%s rewrite_used=%s "
                "confidence=%.2f subquery_count=%d step_back=%s "
                "fallback=%s constraints=%d history_used=%s latency_ms=%d",
                plan.query_mode.value,
                plan.rewrite_strategy,
                plan.rewritten_query is not None,
                plan.confidence,
                len(plan.subqueries),
                plan.step_back_query is not None,
                plan.fallback_used,
                len(constraints),
                plan.used_history,
                plan.latency_ms,
            )

            return plan

        except asyncio.TimeoutError:
            latency = int((time.perf_counter() - t0) * 1000)
            logger.warning(
                "query_rewrite.timeout latency_ms=%d timeout_s=%.1f",
                latency, self._timeout,
            )
            return RetrievalPlan(
                original_query=query_text,
                query_mode=QueryMode.DIRECT,
                rewrite_strategy=RewriteStrategy.SAFE_FALLBACK.value,
                confidence=1.0,
                fallback_used=True,
                latency_ms=latency,
                rewrite_reason="timeout_fallback",
            )

        except Exception:
            latency = int((time.perf_counter() - t0) * 1000)
            logger.warning(
                "query_rewrite.error latency_ms=%d",
                latency,
                exc_info=True,
            )
            return RetrievalPlan(
                original_query=query_text,
                query_mode=QueryMode.DIRECT,
                rewrite_strategy=RewriteStrategy.SAFE_FALLBACK.value,
                confidence=1.0,
                fallback_used=True,
                latency_ms=latency,
                rewrite_reason="error_fallback",
            )

    # ── V2: Constraint detection ─────────────────────────────────────

    @staticmethod
    def _detect_constraints(query: str) -> tuple[str, ...]:
        """Detect constraint tokens in query that must be preserved.

        Returns tuple of constraint labels found (e.g., "year", "negation",
        "unit", "role", "contract", "permission", "time_period").

        This is a lightweight heuristic — not a full NER.
        No raw text is returned, only labels.
        """
        labels = []
        label_map = {
            0: "year",
            1: "time_period",
            2: "negation_vi",
            3: "negation_en",
            4: "unit",
            5: "role",
            6: "contract",
            7: "permission_vi",
            8: "permission_en",
        }
        for i, pattern in enumerate(_CONSTRAINT_PATTERNS):
            if pattern.search(query):
                labels.append(label_map.get(i, f"constraint_{i}"))
        return tuple(labels)

    # ── V2: Gating strategy ──────────────────────────────────────────

    def _determine_strategy(
        self,
        query: str,
        mode: QueryMode,
        history: list[Any] | None,
        has_constraints: bool,
    ) -> RewriteStrategy:
        """Determine the rewrite strategy (gating decision).

        Constraints are an orthogonal signal, NOT a query mode.
        They influence gating but don't override the behavior class.

        Rules:
          1. DIRECT + no follow-up markers → NO_REWRITE
          2. FOLLOW_UP + history available → CONTEXTUAL_REWRITE
          3. FOLLOW_UP + no history → SAFE_FALLBACK
          4. AMBIGUOUS + no history → SAFE_FALLBACK
          5. AMBIGUOUS + history → CONTEXTUAL_REWRITE
          6. MULTI_HOP → CONTROLLED_DECOMPOSITION
          7. COMPARISON → CONTROLLED_DECOMPOSITION
          8. OVERVIEW → LIGHT_NORMALIZE (broaden, don't split)
          9. SPECIFIC → LIGHT_NORMALIZE (default conservative)
             escalate to CONTEXTUAL_REWRITE only if follow-up markers present
         10. Any mode + constraints → cap at LIGHT_NORMALIZE max
        """
        q = query.lower().strip()
        has_follow_up = self._has_follow_up_markers(q)
        has_history = bool(history)

        # DIRECT clear queries: no rewrite
        if mode == QueryMode.DIRECT and not has_follow_up:
            return RewriteStrategy.NO_REWRITE

        # FOLLOW_UP: contextual rewrite if history, else fallback
        if mode == QueryMode.FOLLOW_UP or has_follow_up:
            if has_history:
                return RewriteStrategy.CONTEXTUAL_REWRITE
            return RewriteStrategy.SAFE_FALLBACK

        # AMBIGUOUS: contextual if history, else fallback
        if mode == QueryMode.AMBIGUOUS:
            if has_history:
                return RewriteStrategy.CONTEXTUAL_REWRITE
            return RewriteStrategy.SAFE_FALLBACK

        # MULTI_HOP / COMPARISON: controlled decomposition
        # BUT if heavy constraints, cap at LIGHT_NORMALIZE to avoid losing them
        if mode in (QueryMode.MULTI_HOP, QueryMode.COMPARISON):
            if has_constraints:
                return RewriteStrategy.LIGHT_NORMALIZE
            return RewriteStrategy.CONTROLLED_DECOMPOSITION

        # OVERVIEW: light normalize (broaden, don't split)
        if mode == QueryMode.OVERVIEW:
            return RewriteStrategy.LIGHT_NORMALIZE

        # SPECIFIC: default conservative (LIGHT_NORMALIZE)
        # Most specific queries are already clear and self-contained.
        if mode == QueryMode.SPECIFIC:
            return RewriteStrategy.LIGHT_NORMALIZE

        # Default: light normalize (conservative)
        return RewriteStrategy.LIGHT_NORMALIZE

    # ── Classification ───────────────────────────────────────────────

    def _classify_query_mode(
        self, query: str, history: list[Any] | None = None,
    ) -> QueryMode:
        """Classify query intent using keyword heuristics.

        Returns the "shape" / behavior class of the query.
        Constraints are detected separately via _detect_constraints()
        and influence gating, not classification.
        """
        q = query.lower().strip()

        # Follow-up check first (has context dependency)
        if self._has_follow_up_markers(q):
            return QueryMode.FOLLOW_UP

        # Short, clear queries → direct
        if len(q) < _MIN_QUERY_LENGTH_FOR_REWRITE:
            return QueryMode.DIRECT

        # Keyword classification
        if any(k in q for k in _MULTI_HOP_KW):
            return QueryMode.MULTI_HOP
        if any(k in q for k in _COMPARE_KW):
            return QueryMode.COMPARISON
        if any(k in q for k in _OVERVIEW_KW):
            return QueryMode.OVERVIEW
        if any(k in q for k in _SPECIFIC_KW):
            return QueryMode.SPECIFIC

        # Check for multiple question marks or conjunctions → multi_hop
        if q.count("?") > 1 or (" và " in q and "?" in q) or (" and " in q and "?" in q):
            return QueryMode.MULTI_HOP

        # Very short or single-word → ambiguous
        word_count = len(q.split())
        if word_count <= 2:
            return QueryMode.AMBIGUOUS

        return QueryMode.DIRECT

    def _has_follow_up_markers(self, query: str) -> bool:
        """Check if query contains follow-up reference markers."""
        q = query.lower()
        return any(m in q for m in _FOLLOW_UP_MARKERS_VI + _FOLLOW_UP_MARKERS_EN)

    # ── History resolution ───────────────────────────────────────────

    def _resolve_history_references(
        self, query: str, history: list[Any],
    ) -> str | None:
        """
        Resolve pronoun/reference in query using recent history.

        Strategy:
          1. Only look at last _MAX_HISTORY_TURNS turns
          2. Find the most recent user turn with a salient topic
          3. Only resolve if query contains clear pronoun references
          4. Extract a short subject from the history topic
          5. Replace the pronoun with the subject to build a standalone query
          6. If topic is too long or unclear, return None (no guess)

        Returns resolved query or None if cannot resolve confidently.
        """
        if not history:
            return None

        # Find last user turn (recent only)
        recent = history[-_MAX_HISTORY_TURNS:]
        last_user_text = None
        for turn in reversed(recent):
            role = getattr(turn, "role", None) or (turn.get("role") if isinstance(turn, dict) else None)
            text = getattr(turn, "text", None) or (turn.get("text") if isinstance(turn, dict) else None)
            if role == "user" and text and text.strip():
                last_user_text = text.strip()
                break

        if not last_user_text:
            return None

        # Only resolve if query has clear pronoun/reference markers
        q_lower = query.lower()

        # Map of pronoun markers to check
        _PRONOUN_MARKERS_VI = ("cái này", "cái đó", "nó có", "nó")
        _REFERENCE_MARKERS_VI = ("ở trên", "nói ở trên", "vừa nói")
        # English pronouns: use word-boundary regex to avoid false
        # positives (e.g. "item" matching "it", "with" matching "th")
        _PRONOUN_RE_EN = re.compile(r"\b(?:this|that|it)\b", re.IGNORECASE)

        has_pronoun_vi = any(m in q_lower for m in _PRONOUN_MARKERS_VI)
        has_reference_vi = any(m in q_lower for m in _REFERENCE_MARKERS_VI)
        has_pronoun_en = bool(_PRONOUN_RE_EN.search(q_lower))

        if not (has_pronoun_vi or has_reference_vi or has_pronoun_en):
            return None

        # Topic too long or unclear → don't resolve (no guessing)
        if len(last_user_text) > _MAX_HISTORY_TEXT_LEN:
            return None

        # Extract salient subject: first sentence, capped at 80 chars
        # This avoids dumping the entire user turn into the query
        subject = last_user_text
        # Split on sentence boundaries
        for sep in (".", "?", "!", "\n"):
            if sep in subject:
                subject = subject[:subject.index(sep)]
                break
        subject = subject.strip()[:80].strip()

        if not subject or len(subject) < 3:
            return None

        # Build standalone query by substituting pronoun with subject
        resolved = query
        # Vietnamese pronoun substitution
        for marker in ("cái này", "cái đó"):
            if marker in resolved.lower():
                idx = resolved.lower().index(marker)
                resolved = resolved[:idx] + subject + resolved[idx + len(marker):]
                return resolved

        # Vietnamese reference markers → prepend subject
        for marker in ("ở trên", "nói ở trên", "vừa nói"):
            if marker in resolved.lower():
                return f"{subject}: {query}"

        # "nó" substitution (Vietnamese) — careful, only standalone "nó"
        if has_pronoun_vi and "nó" in q_lower:
            idx = resolved.lower().index("nó")
            # Only replace if "nó" is at word boundary
            before = resolved[idx - 1] if idx > 0 else " "
            after = resolved[idx + 2] if idx + 2 < len(resolved) else " "
            if not before.isalpha() and not after.isalpha():
                resolved = resolved[:idx] + subject + resolved[idx + 2:]
                return resolved

        # English pronoun substitution
        for marker in ("this", "that", "it"):
            if marker in q_lower:
                idx = q_lower.index(marker)
                before = resolved[idx - 1] if idx > 0 else " "
                after = resolved[idx + len(marker)] if idx + len(marker) < len(resolved) else " "
                if not before.isalpha() and not after.isalpha():
                    resolved = resolved[:idx] + subject + resolved[idx + len(marker):]
                    return resolved

        # Fallback: prepend subject as context (still better than raw append)
        return f"{subject} — {query}"

    # ── LLM Rewrite ──────────────────────────────────────────────────

    _SYSTEM_PROMPT = """\
You are a query rewriting assistant for a document search system.
Your job is to improve search queries for better document retrieval.

Rules:
- Preserve the original intent exactly — do NOT change the meaning
- Make implicit references explicit
- Use domain-appropriate terminology when possible
- If query is already clear, return it unchanged with high confidence
- Do NOT invent filters, metadata, or facts
- Do NOT hallucinate document names or content
- Do NOT add filter-like tokens such as tenant_id:, doc_id:, source:, metadata:, tag:, filter:
- Keep rewrites concise (under 200 chars)
- Respond in the SAME language as the input query

CONSTRAINT PRESERVATION (CRITICAL):
- NEVER drop year/date/time constraints (e.g., "năm 2025", "2024", "quý 3")
- NEVER drop entity/role constraints (e.g., "giảng viên hợp đồng", "sinh viên")
- NEVER drop negation (e.g., "không", "chưa", "not", "without")
- NEVER drop unit/department scope (e.g., "cấp khoa", "phòng nhân sự")
- NEVER drop conditions/permissions (e.g., "ai được", "điều kiện", "bắt buộc")
- If you cannot preserve all constraints, return the original query unchanged

DECOMPOSITION RULES:
- Only split into subqueries when the query asks about multiple DISTINCT aspects
- Maximum 2 subqueries
- Each subquery must preserve ALL constraints from the original
- Do NOT split comparison queries — keep them as one query
- Do NOT split overview queries — broaden instead

Output JSON only:
{
  "rewritten_query": "improved query or null if original is good enough",
  "step_back_query": "broader retrieval query or null",
  "subqueries": ["sub1", "sub2"],
  "confidence": 0.8,
  "reason": "brief explanation"
}
"""

    async def _llm_rewrite(
        self,
        query: str,
        mode: QueryMode,
        history: list[Any] | None,
        *,
        constraints: tuple[str, ...] = (),
    ) -> dict | None:
        """Call LLM for query rewrite. Returns parsed JSON or None."""
        api_key = getattr(settings, "OPENAI_API_KEY", "")
        if not api_key:
            return None

        # Build user prompt
        user_content = f"Query mode: {mode.value}\nQuery: {query[:self._max_query_chars]}"

        # V2: include constraint info for the LLM
        if constraints:
            user_content += f"\nDetected constraints (MUST PRESERVE): {', '.join(constraints)}"

        if history:
            recent = history[-_MAX_HISTORY_TURNS:]
            history_lines = []
            for turn in recent:
                role = getattr(turn, "role", None) or (turn.get("role") if isinstance(turn, dict) else None)
                text = getattr(turn, "text", None) or (turn.get("text") if isinstance(turn, dict) else None)
                if role and text:
                    history_lines.append(f"{role}: {text[:_MAX_HISTORY_TEXT_LEN]}")
            if history_lines:
                user_content += f"\n\nRecent conversation:\n" + "\n".join(history_lines)

        payload = {
            "model": self._model,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": self._SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        }

        async with httpx.AsyncClient(timeout=self._timeout + 1.0) as client:
            resp = await asyncio.wait_for(
                client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                ),
                timeout=self._timeout,
            )
            resp.raise_for_status()
            data = resp.json()

        try:
            text = data["choices"][0]["message"]["content"]
            result = json.loads(text)
        except Exception:
            logger.warning("query_rewrite.llm_parse_error")
            return None

        # Validate structure
        if not isinstance(result, dict):
            return None

        return result

    # ── Guardrails ───────────────────────────────────────────────────

    def _apply_guardrails(
        self,
        *,
        original_query: str,
        rewritten_query: str | None,
        step_back_query: str | None,
        subqueries: tuple[str, ...],
        query_mode: QueryMode,
        rewrite_strategy: RewriteStrategy,
        confidence: float,
        reason: str,
        used_history: bool,
        constraints: tuple[str, ...] = (),
        latency_ms: int,
    ) -> RetrievalPlan:
        """
        Apply safety guardrails to the rewrite output.

        Guardrails:
          - Clamp confidence 0..1
          - Low-confidence → strip all rewrites
          - Validate each query (_is_valid_query)
          - Reject hallucinated filter tokens
          - V2: Constraint preservation check
          - Full cross-dedupe between all queries
          - Trim length
          - Cap subqueries at max
        """

        # Clamp confidence
        confidence = max(0.0, min(1.0, confidence))

        # Reject low-confidence rewrites
        fallback = False
        if confidence < self._confidence_threshold:
            rewritten_query = None
            subqueries = ()
            step_back_query = None
            fallback = True
            reason = f"low_confidence({confidence:.2f})"
            rewrite_strategy = RewriteStrategy.SAFE_FALLBACK

        # V2: Constraint preservation check
        if rewritten_query and constraints:
            if not self._constraints_preserved(original_query, rewritten_query):
                # Constraints were lost in rewrite — reject
                rewritten_query = None
                reason = "constraint_preservation_rejected"

        # V2: step_back is inherently broad — drop it when constraints present
        # to avoid retrieval dilution from a too-wide query
        if step_back_query and constraints:
            step_back_query = None

        # ── Sanitize individual queries ──────────────────────────────

        def _sanitize(q: str | None) -> str | None:
            """Validate, trim, and reject invalid queries."""
            if not q:
                return None
            q = q.strip()[:self._max_query_chars]
            if not self._is_valid_query(q):
                return None
            return q

        rewritten_query = _sanitize(rewritten_query)
        step_back_query = _sanitize(step_back_query)

        # ── Cross-dedupe ─────────────────────────────────────────────
        # Track normalized versions to prevent duplicates
        seen: set[str] = set()
        orig_norm = " ".join(original_query.lower().split())
        seen.add(orig_norm)

        def _dedupe(q: str | None) -> str | None:
            if not q:
                return None
            norm = " ".join(q.lower().split())
            if norm in seen:
                return None
            seen.add(norm)
            return q

        rewritten_query = _dedupe(rewritten_query)
        step_back_query = _dedupe(step_back_query)

        # Sanitize and dedupe subqueries (filter first, then cap)
        clean_subs: list[str] = []
        for sq in subqueries:
            sq_clean = _sanitize(sq)
            sq_clean = _dedupe(sq_clean)
            # V2: each subquery must also preserve constraints
            if sq_clean and constraints:
                if not self._constraints_preserved(original_query, sq_clean):
                    continue
            if sq_clean:
                clean_subs.append(sq_clean)
            if len(clean_subs) >= self._max_subqueries:
                break
        subqueries = tuple(clean_subs)

        return RetrievalPlan(
            original_query=original_query,
            query_mode=query_mode,
            rewrite_strategy=rewrite_strategy.value,
            rewritten_query=rewritten_query,
            step_back_query=step_back_query,
            subqueries=subqueries,
            confidence=confidence,
            rewrite_reason=reason,
            used_history=used_history,
            latency_ms=latency_ms,
            fallback_used=fallback,
        )

    # ── V2: Constraint preservation check ────────────────────────────

    @staticmethod
    def _constraints_preserved(
        original: str, rewritten: str,
    ) -> bool:
        """Check if critical constraints from original are still in rewritten.

        Currently verified:
          - Year/date numbers (2024, 2025, etc.)
          - Negation words (Vietnamese + English)

        NOT yet verified (detected for gating only):
          - Role / unit / contract / permission keywords

        Lightweight heuristic — not exhaustive.
        Favors false positives (allow) over false negatives (reject).
        """
        orig_lower = original.lower()
        rewrite_lower = rewritten.lower()

        # Check year preservation
        years = re.findall(r"\b20\d{2}\b", orig_lower)
        for year in years:
            if year not in rewrite_lower:
                return False

        # Check negation preservation
        negations = [
            n for n in ("không", "chưa", "không được", "not", "never", "without")
            if n in orig_lower
        ]
        for neg in negations:
            if neg not in rewrite_lower:
                return False

        return True
