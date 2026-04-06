"""
Phase 1.2B — LLM judge graders.

Provides two core semantic judges:
  1. faithfulness_judge  — Is the answer grounded in cited evidence?
  2. answer_relevance_judge — Does the answer actually address the question?

Design rules:
  - Judges are SUPPLEMENTARY to 1.2A deterministic graders, never replace.
  - Context sent to judge is MINIMAL: cited snippets, not full retrieval set.
  - Output is structured JSON, fail-safe parsed.
  - Caching by (case_id, answer_hash, judge_type, model) to avoid re-grading.
  - Temperature = 0 for determinism.
  - Never crashes the run on judge failure — marks judge_failed and continues.
  - No judge runs unless user explicitly enables with --with-judges.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Judge output schema ──────────────────────────────────────────────────────


@dataclass
class JudgeResult:
    """Structured output from an LLM judge."""

    judge_type: str  # "faithfulness" | "answer_relevance"
    score: int = 0  # 0..5
    verdict: str = ""  # short string
    rationale: str = ""  # short string
    failure_tags: list[str] = field(default_factory=list)

    # Metadata
    judge_model: str = ""
    latency_ms: int = 0
    cached: bool = False
    judge_failed: bool = False
    error: str | None = None


# ── Cache ─────────────────────────────────────────────────────────────────────

_cache: dict[str, JudgeResult] = {}


def _cache_key(
    case_id: str,
    answer_text: str | None,
    judge_type: str,
    model: str,
) -> str:
    """Build a deterministic cache key."""
    answer_hash = hashlib.sha256(
        (answer_text or "").encode("utf-8")
    ).hexdigest()[:16]
    return f"{case_id}:{answer_hash}:{judge_type}:{model}"


def clear_cache() -> None:
    """Clear the in-memory judge cache."""
    _cache.clear()


# ── Judge config ──────────────────────────────────────────────────────────────


@dataclass
class JudgeConfig:
    """Configuration for LLM judge execution."""

    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 300
    timeout_s: float = 15.0
    max_retries: int = 1
    max_snippet_chars: int = 400
    max_history_chars: int = 300
    max_snippets: int = 3


DEFAULT_JUDGE_CONFIG = JudgeConfig()


# ── Prompt builders ───────────────────────────────────────────────────────────


def _truncate(text: str | None, max_chars: int) -> str:
    """Safely truncate text."""
    if not text:
        return ""
    text = text.strip()
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text


def _build_snippets_block(
    citations: list[dict] | None,
    config: JudgeConfig,
) -> str:
    """Build a compact cited-snippets block for the judge prompt."""
    if not citations:
        return "(no citations provided)"

    parts: list[str] = []
    for i, c in enumerate(citations[:config.max_snippets], start=1):
        snippet = _truncate(
            c.get("snippet") or c.get("text") or "",
            config.max_snippet_chars,
        )
        doc_id = c.get("document_id") or c.get("source_document_id") or "?"
        if snippet:
            parts.append(f"[{i}] doc={doc_id}: {snippet}")

    return "\n".join(parts) if parts else "(no snippet content)"


def _build_history_block(
    history: list[dict] | None,
    config: JudgeConfig,
) -> str:
    """Build compact conversation history for judge context."""
    if not history:
        return ""

    lines: list[str] = []
    total_chars = 0
    for turn in history[-4:]:  # Last 4 turns max
        role = turn.get("role", "?")
        text = _truncate(turn.get("text", ""), 150)
        line = f"{role}: {text}"
        total_chars += len(line)
        if total_chars > config.max_history_chars:
            break
        lines.append(line)

    return "\n".join(lines)


# ── Faithfulness judge ────────────────────────────────────────────────────────


_FAITHFULNESS_SYSTEM = """You are an evaluation judge assessing FAITHFULNESS of an AI assistant's answer.

Faithfulness = the answer is grounded in the provided evidence. It does NOT hallucinate facts not present in the citations.

Score 0-5:
0 = Completely hallucinated, no connection to evidence
1 = Mostly hallucinated with minor grounding
2 = Mixed: some grounded, some hallucinated claims
3 = Mostly grounded but contains minor unsupported claims
4 = Well grounded with very minor extrapolations
5 = Fully grounded in the cited evidence

Respond ONLY with this exact JSON format:
{"score": <0-5>, "verdict": "<one phrase>", "rationale": "<1-2 sentences>", "failure_tags": []}

failure_tags options: ["hallucination", "unsupported_claim", "fabricated_detail", "contradicts_evidence"]
Use empty list if no failures detected."""


def _faithfulness_user_prompt(
    question: str,
    answer_text: str,
    snippets_block: str,
    expected_answer_type: str,
    history_block: str = "",
) -> str:
    parts = [
        f"Question: {_truncate(question, 500)}",
        f"Expected answer type: {expected_answer_type}",
    ]
    if history_block:
        parts.append(f"Conversation history:\n{history_block}")
    parts.append(f"Cited evidence:\n{snippets_block}")
    parts.append(f"Assistant answer:\n{_truncate(answer_text, 1000)}")
    parts.append("Evaluate the faithfulness of this answer to the cited evidence.")
    return "\n\n".join(parts)


# ── Answer relevance judge ────────────────────────────────────────────────────


_RELEVANCE_SYSTEM = """You are an evaluation judge assessing ANSWER RELEVANCE of an AI assistant's answer.

Answer Relevance = the answer directly addresses the user's question. It is not generic, off-topic, or evasive.

Score 0-5:
0 = Completely irrelevant or off-topic
1 = Barely related, mostly generic
2 = Partially relevant but misses the core question
3 = Relevant but too generic or missing key details
4 = Good relevance, addresses the question well
5 = Perfectly relevant, directly and specifically answers the question

Respond ONLY with this exact JSON format:
{"score": <0-5>, "verdict": "<one phrase>", "rationale": "<1-2 sentences>", "failure_tags": []}

failure_tags options: ["off_topic", "too_generic", "evasive", "wrong_focus", "partial_answer"]
Use empty list if no failures detected."""


def _relevance_user_prompt(
    question: str,
    answer_text: str,
    expected_answer_type: str,
    expected_keywords: list[str] | None = None,
    history_block: str = "",
) -> str:
    parts = [
        f"Question: {_truncate(question, 500)}",
        f"Expected answer type: {expected_answer_type}",
    ]
    if expected_keywords:
        parts.append(f"Expected keywords: {', '.join(expected_keywords[:10])}")
    if history_block:
        parts.append(f"Conversation history:\n{history_block}")
    parts.append(f"Assistant answer:\n{_truncate(answer_text, 1000)}")
    parts.append("Evaluate how relevant this answer is to the question asked.")
    return "\n\n".join(parts)


# ── JSON parser (robust) ─────────────────────────────────────────────────────


def _parse_judge_json(raw: str, judge_type: str) -> JudgeResult | None:
    """Parse judge output JSON. Returns None if unparseable."""
    if not raw:
        return None

    # Try to find JSON in the response
    text = raw.strip()

    # Sometimes the model wraps in ```json ... ```
    if "```json" in text:
        start = text.index("```json") + 7
        end = text.index("```", start) if "```" in text[start:] else len(text)
        text = text[start:end].strip()
    elif "```" in text:
        start = text.index("```") + 3
        end = text.index("```", start) if "```" in text[start:] else len(text)
        text = text[start:end].strip()

    # Find the first { and last }
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start >= 0 and brace_end > brace_start:
        text = text[brace_start:brace_end + 1]

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None

    return JudgeResult(
        judge_type=judge_type,
        score=max(0, min(5, int(data.get("score", 0)))),
        verdict=str(data.get("verdict", ""))[:200],
        rationale=str(data.get("rationale", ""))[:500],
        failure_tags=list(data.get("failure_tags", []) or []),
    )


# ── LLM call (via OpenAI) ────────────────────────────────────────────────────


async def _call_llm(
    system_prompt: str,
    user_prompt: str,
    config: JudgeConfig,
) -> str | None:
    """Call OpenAI chat completions. Returns raw text or None."""
    import httpx

    api_key = ""
    try:
        from app.core.config import settings
        api_key = getattr(settings, "OPENAI_API_KEY", "")
    except Exception:
        import os
        api_key = os.environ.get("OPENAI_API_KEY", "")

    if not api_key:
        return None

    payload = {
        "model": config.model,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    async with httpx.AsyncClient(timeout=config.timeout_s + 2.0) as client:
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
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        return None


# ── Public judge functions ────────────────────────────────────────────────────


async def faithfulness_judge(
    *,
    case_id: str,
    question: str,
    answer_text: str | None,
    citations: list[dict] | None = None,
    history: list[dict] | None = None,
    expected_answer_type: str = "specific",
    config: JudgeConfig | None = None,
) -> JudgeResult:
    """Judge faithfulness of an answer to its cited evidence.

    Fail-safe: returns JudgeResult with judge_failed=True on any error.
    """
    cfg = config or DEFAULT_JUDGE_CONFIG

    # Skip if no answer
    if not answer_text or not answer_text.strip():
        return JudgeResult(
            judge_type="faithfulness",
            score=0,
            verdict="no_answer",
            rationale="No answer text to evaluate",
            judge_model=cfg.model,
        )

    # Check cache
    key = _cache_key(case_id, answer_text, "faithfulness", cfg.model)
    if key in _cache:
        cached = _cache[key]
        cached.cached = True
        return cached

    # Build prompt
    snippets_block = _build_snippets_block(citations, cfg)
    history_block = _build_history_block(history, cfg)
    user_prompt = _faithfulness_user_prompt(
        question, answer_text, snippets_block,
        expected_answer_type, history_block,
    )

    # Call LLM
    t0 = time.perf_counter()
    result = JudgeResult(
        judge_type="faithfulness",
        judge_model=cfg.model,
    )

    for attempt in range(cfg.max_retries + 1):
        try:
            raw = await _call_llm(_FAITHFULNESS_SYSTEM, user_prompt, cfg)
            result.latency_ms = int((time.perf_counter() - t0) * 1000)

            if raw is None:
                result.judge_failed = True
                result.error = "LLM returned None (no API key?)"
                break

            parsed = _parse_judge_json(raw, "faithfulness")
            if parsed:
                parsed.judge_model = cfg.model
                parsed.latency_ms = result.latency_ms
                _cache[key] = parsed
                return parsed

            if attempt < cfg.max_retries:
                logger.debug("Judge parse failed, retry %d", attempt + 1)
                continue

            result.judge_failed = True
            result.error = f"JSON parse failed after {cfg.max_retries + 1} attempts"

        except Exception as exc:
            result.latency_ms = int((time.perf_counter() - t0) * 1000)
            result.judge_failed = True
            result.error = f"{exc.__class__.__name__}: {exc}"
            break

    return result


async def answer_relevance_judge(
    *,
    case_id: str,
    question: str,
    answer_text: str | None,
    history: list[dict] | None = None,
    expected_answer_type: str = "specific",
    expected_keywords: list[str] | None = None,
    config: JudgeConfig | None = None,
) -> JudgeResult:
    """Judge answer relevance to the question asked.

    Fail-safe: returns JudgeResult with judge_failed=True on any error.
    """
    cfg = config or DEFAULT_JUDGE_CONFIG

    # Skip if no answer
    if not answer_text or not answer_text.strip():
        return JudgeResult(
            judge_type="answer_relevance",
            score=0,
            verdict="no_answer",
            rationale="No answer text to evaluate",
            judge_model=cfg.model,
        )

    # Check cache
    key = _cache_key(case_id, answer_text, "answer_relevance", cfg.model)
    if key in _cache:
        cached = _cache[key]
        cached.cached = True
        return cached

    # Build prompt
    history_block = _build_history_block(history, cfg)
    user_prompt = _relevance_user_prompt(
        question, answer_text, expected_answer_type,
        expected_keywords, history_block,
    )

    # Call LLM
    t0 = time.perf_counter()
    result = JudgeResult(
        judge_type="answer_relevance",
        judge_model=cfg.model,
    )

    for attempt in range(cfg.max_retries + 1):
        try:
            raw = await _call_llm(_RELEVANCE_SYSTEM, user_prompt, cfg)
            result.latency_ms = int((time.perf_counter() - t0) * 1000)

            if raw is None:
                result.judge_failed = True
                result.error = "LLM returned None (no API key?)"
                break

            parsed = _parse_judge_json(raw, "answer_relevance")
            if parsed:
                parsed.judge_model = cfg.model
                parsed.latency_ms = result.latency_ms
                _cache[key] = parsed
                return parsed

            if attempt < cfg.max_retries:
                logger.debug("Judge parse failed, retry %d", attempt + 1)
                continue

            result.judge_failed = True
            result.error = f"JSON parse failed after {cfg.max_retries + 1} attempts"

        except Exception as exc:
            result.latency_ms = int((time.perf_counter() - t0) * 1000)
            result.judge_failed = True
            result.error = f"{exc.__class__.__name__}: {exc}"
            break

    return result


# ── Batch runner ──────────────────────────────────────────────────────────────


async def run_judges_for_case(
    *,
    case_id: str,
    question: str,
    answer_text: str | None,
    citations: list[dict] | None = None,
    history: list[dict] | None = None,
    expected_answer_type: str = "specific",
    expected_keywords: list[str] | None = None,
    config: JudgeConfig | None = None,
) -> dict[str, JudgeResult]:
    """Run all enabled judges for a single case.

    Returns dict mapping judge_type -> JudgeResult.
    """
    results: dict[str, JudgeResult] = {}

    results["faithfulness"] = await faithfulness_judge(
        case_id=case_id,
        question=question,
        answer_text=answer_text,
        citations=citations,
        history=history,
        expected_answer_type=expected_answer_type,
        config=config,
    )

    results["answer_relevance"] = await answer_relevance_judge(
        case_id=case_id,
        question=question,
        answer_text=answer_text,
        history=history,
        expected_answer_type=expected_answer_type,
        expected_keywords=expected_keywords,
        config=config,
    )

    return results
