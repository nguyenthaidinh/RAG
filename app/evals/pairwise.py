"""
Phase 1.2B — Pairwise regression comparison.

Compares baseline (A) vs candidate (B) outputs on the same eval cases
to detect improvement or regression after a code/model change.

Design rules:
  - Pairwise only runs when user explicitly requests --mode pairwise.
  - Uses LLM judge to compare A vs B side-by-side.
  - Output: winner (A/B/tie), rationale, failure_tags.
  - Aggregation: win_rate, loss_rate, tie_rate per tenant/slice/type.
  - Context is minimal: question + answers + short cited snippets.
  - Fail-safe: judge failure -> tie with error noted.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# ── Pairwise result schema ────────────────────────────────────────────────────


@dataclass
class PairwiseResult:
    """Result of a single pairwise comparison."""

    case_id: str
    tenant_id: str
    expected_answer_type: str

    winner: str = "tie"  # "A" | "B" | "tie"
    rationale: str = ""
    failure_tags: list[str] = field(default_factory=list)

    # Metadata
    judge_model: str = ""
    latency_ms: int = 0
    judge_failed: bool = False
    error: str | None = None

    # Context (for reporting)
    question: str = ""
    answer_a_preview: str = ""
    answer_b_preview: str = ""
    slice_tags: list[str] = field(default_factory=list)


@dataclass
class PairwiseAggregation:
    """Aggregated pairwise comparison results."""

    total: int = 0
    wins_a: int = 0
    wins_b: int = 0
    ties: int = 0
    judge_failures: int = 0

    @property
    def win_rate_a(self) -> float:
        return self.wins_a / self.total if self.total else 0.0

    @property
    def win_rate_b(self) -> float:
        return self.wins_b / self.total if self.total else 0.0

    @property
    def tie_rate(self) -> float:
        return self.ties / self.total if self.total else 0.0

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "wins_a": self.wins_a,
            "wins_b": self.wins_b,
            "ties": self.ties,
            "win_rate_a": round(self.win_rate_a, 4),
            "win_rate_b": round(self.win_rate_b, 4),
            "tie_rate": round(self.tie_rate, 4),
            "judge_failures": self.judge_failures,
        }


# ── Pairwise judge prompt ────────────────────────────────────────────────────


_PAIRWISE_SYSTEM = """You are an evaluation judge comparing two AI assistant answers (A and B) to the same question.

Decide which answer is BETTER based on these criteria (in priority order):
1. More grounded in evidence (less hallucination)
2. More directly answers the question
3. Less generic / more specific
4. Better citation support
5. More natural and clear

Respond ONLY with this exact JSON format:
{"winner": "A" or "B" or "tie", "rationale": "<1-2 sentences>", "failure_tags": []}

failure_tags options: ["a_hallucination", "b_hallucination", "a_off_topic", "b_off_topic", "a_too_generic", "b_too_generic", "both_poor", "both_good"]
Use empty list if no specific failures noted.

IMPORTANT:
- If both answers are equally good or equally poor, choose "tie".
- Judge based on quality, not length.
- Do not be biased toward longer answers."""


def _build_pairwise_prompt(
    question: str,
    answer_a: str,
    answer_b: str,
    citations_a: list[dict] | None = None,
    citations_b: list[dict] | None = None,
    expected_answer_type: str = "specific",
    history: list[dict] | None = None,
    max_answer_chars: int = 800,
    max_snippet_chars: int = 300,
) -> str:
    """Build the user prompt for pairwise comparison."""
    from app.evals.graders.llm_judge import _truncate

    parts = [
        f"Question: {_truncate(question, 500)}",
        f"Expected answer type: {expected_answer_type}",
    ]

    if history:
        hist_lines = []
        for t in history[-3:]:
            role = t.get("role", "?")
            text = _truncate(t.get("text", ""), 120)
            hist_lines.append(f"  {role}: {text}")
        parts.append(f"Conversation history:\n" + "\n".join(hist_lines))

    # Answer A
    parts.append(f"Answer A:\n{_truncate(answer_a, max_answer_chars)}")
    if citations_a:
        snips_a = []
        for c in citations_a[:3]:
            s = _truncate(c.get("snippet", ""), max_snippet_chars)
            if s:
                snips_a.append(f"  - {s}")
        if snips_a:
            parts.append(f"Citations A:\n" + "\n".join(snips_a))

    # Answer B
    parts.append(f"Answer B:\n{_truncate(answer_b, max_answer_chars)}")
    if citations_b:
        snips_b = []
        for c in citations_b[:3]:
            s = _truncate(c.get("snippet", ""), max_snippet_chars)
            if s:
                snips_b.append(f"  - {s}")
        if snips_b:
            parts.append(f"Citations B:\n" + "\n".join(snips_b))

    parts.append("Which answer is better? Respond with JSON only.")
    return "\n\n".join(parts)


# ── Parser ────────────────────────────────────────────────────────────────────


def _parse_pairwise_json(raw: str) -> dict | None:
    """Parse pairwise judge JSON. Returns None if unparseable."""
    if not raw:
        return None

    text = raw.strip()

    # Handle markdown code blocks
    if "```json" in text:
        start = text.index("```json") + 7
        end = text.index("```", start) if "```" in text[start:] else len(text)
        text = text[start:end].strip()
    elif "```" in text:
        start = text.index("```") + 3
        end = text.index("```", start) if "```" in text[start:] else len(text)
        text = text[start:end].strip()

    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start >= 0 and brace_end > brace_start:
        text = text[brace_start:brace_end + 1]

    try:
        data = json.loads(text)
        winner = str(data.get("winner", "tie")).upper()
        if winner not in ("A", "B", "TIE"):
            winner = "tie"
        return {
            "winner": winner.lower() if winner == "TIE" else winner,
            "rationale": str(data.get("rationale", ""))[:500],
            "failure_tags": list(data.get("failure_tags", []) or []),
        }
    except json.JSONDecodeError:
        return None


# ── Pairwise judge execution ─────────────────────────────────────────────────


async def pairwise_judge(
    *,
    case_id: str,
    tenant_id: str,
    question: str,
    answer_a: str,
    answer_b: str,
    citations_a: list[dict] | None = None,
    citations_b: list[dict] | None = None,
    expected_answer_type: str = "specific",
    history: list[dict] | None = None,
    slice_tags: list[str] | None = None,
    model: str = "gpt-4o-mini",
    timeout_s: float = 15.0,
) -> PairwiseResult:
    """Compare two answers for the same question.

    Fail-safe: returns tie with judge_failed=True on any error.
    """
    result = PairwiseResult(
        case_id=case_id,
        tenant_id=tenant_id,
        expected_answer_type=expected_answer_type,
        judge_model=model,
        question=question[:100],
        answer_a_preview=(answer_a or "")[:120],
        answer_b_preview=(answer_b or "")[:120],
        slice_tags=slice_tags or [],
    )

    # Edge cases
    if not answer_a and not answer_b:
        result.winner = "tie"
        result.rationale = "Both answers are empty"
        return result
    if not answer_a:
        result.winner = "B"
        result.rationale = "Answer A is empty"
        return result
    if not answer_b:
        result.winner = "A"
        result.rationale = "Answer B is empty"
        return result

    # Build prompt
    user_prompt = _build_pairwise_prompt(
        question, answer_a, answer_b,
        citations_a, citations_b,
        expected_answer_type, history,
    )

    # Call LLM
    t0 = time.perf_counter()
    from app.evals.graders.llm_judge import JudgeConfig, _call_llm

    cfg = JudgeConfig(model=model, timeout_s=timeout_s)

    max_retries = 1
    for attempt in range(max_retries + 1):
        try:
            raw = await _call_llm(_PAIRWISE_SYSTEM, user_prompt, cfg)
            result.latency_ms = int((time.perf_counter() - t0) * 1000)

            if raw is None:
                result.judge_failed = True
                result.error = "LLM returned None (no API key?)"
                result.winner = "tie"
                break

            parsed = _parse_pairwise_json(raw)
            if parsed:
                result.winner = parsed["winner"]
                result.rationale = parsed["rationale"]
                result.failure_tags = parsed["failure_tags"]
                return result

            if attempt < max_retries:
                continue

            result.judge_failed = True
            result.error = "JSON parse failed"
            result.winner = "tie"

        except Exception as exc:
            result.latency_ms = int((time.perf_counter() - t0) * 1000)
            result.judge_failed = True
            result.error = f"{exc.__class__.__name__}: {exc}"
            result.winner = "tie"
            break

    return result


# ── Load per_case_results JSONL ───────────────────────────────────────────────


def load_per_case_results(path: str | Path) -> list[dict]:
    """Load a per_case_results.jsonl file as a list of dicts.

    Used to load baseline/candidate outputs for pairwise comparison.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")

    results: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def align_results(
    baseline: list[dict],
    candidate: list[dict],
) -> list[tuple[dict, dict]]:
    """Align baseline and candidate results by case_id.

    Returns list of (baseline_case, candidate_case) tuples.
    Only cases present in BOTH sets are returned.
    """
    baseline_map = {r["case_id"]: r for r in baseline}
    candidate_map = {r["case_id"]: r for r in candidate}

    common_ids = set(baseline_map.keys()) & set(candidate_map.keys())
    pairs = [
        (baseline_map[cid], candidate_map[cid])
        for cid in sorted(common_ids)
    ]

    if len(baseline) != len(pairs) or len(candidate) != len(pairs):
        logger.warning(
            "Pairwise alignment: baseline=%d, candidate=%d, aligned=%d",
            len(baseline), len(candidate), len(pairs),
        )

    return pairs


# ── Aggregation ───────────────────────────────────────────────────────────────


def aggregate_pairwise(results: list[PairwiseResult]) -> PairwiseAggregation:
    """Aggregate pairwise results."""
    agg = PairwiseAggregation(total=len(results))
    for r in results:
        if r.judge_failed:
            agg.judge_failures += 1
            agg.ties += 1  # Count failures as ties
        elif r.winner == "A":
            agg.wins_a += 1
        elif r.winner == "B":
            agg.wins_b += 1
        else:
            agg.ties += 1
    return agg


def aggregate_pairwise_by_group(
    results: list[PairwiseResult],
    key_fn,
) -> dict[str, PairwiseAggregation]:
    """Aggregate pairwise results grouped by arbitrary key."""
    groups: dict[str, list[PairwiseResult]] = defaultdict(list)
    for r in results:
        groups[key_fn(r)].append(r)
    return {k: aggregate_pairwise(v) for k, v in sorted(groups.items())}
