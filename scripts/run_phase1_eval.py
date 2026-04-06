"""
Phase 1 — Offline evaluation runner.

Reads golden_set.jsonl, runs each case through the AssistantService pipeline,
computes retrieval + answer quality metrics, and outputs results as JSON.

Usage:
    python scripts/run_phase1_eval.py
    python scripts/run_phase1_eval.py --golden evals/phase1/golden_set.jsonl --output evals/phase1/results.json
    python scripts/run_phase1_eval.py --tenant demo  # filter to specific tenant

Requirements:
    - Server must be running (or use --direct for in-process execution)
    - golden_set.jsonl must exist with valid cases

Metrics computed:
    Retrieval:  retrieval_hit_at_1, retrieval_hit_at_3, retrieval_hit_at_5
    Answer:     has_answer, has_citations, citation_hit, citation_precision,
                citation_recall, keyword_coverage, no_answer_correctness
    Stability:  latency_ms, answer_source (llm/evidence_fallback/deterministic)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("DATABASE_URL", os.environ.get("DATABASE_URL", ""))
os.environ.setdefault("JWT_SECRET", os.environ.get("JWT_SECRET", "eval-runner"))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("eval_runner")
logger.setLevel(logging.INFO)


# ── Data models ───────────────────────────────────────────────────────


@dataclass
class EvalCase:
    case_id: str
    tenant_id: str
    question: str
    history: list[dict]
    expected_source_document_ids: list[int]
    expected_keywords: list[str]
    answer_type: str
    notes: str = ""


@dataclass
class EvalResult:
    case_id: str
    tenant_id: str
    answer_type: str

    # Answer metrics
    has_answer: bool = False
    answer_text: str | None = None
    answer_source: str = "none"
    intent_detected: str = "none"

    # Citation metrics
    has_citations: bool = False
    citation_count: int = 0
    citation_document_ids: list[int] = field(default_factory=list)
    citation_source_document_ids: list[int] = field(default_factory=list)
    citation_hit: bool = False
    citation_precision: float = 0.0
    citation_recall: float = 0.0

    # Retrieval hit metrics
    retrieval_hit_at_1: bool = False
    retrieval_hit_at_3: bool = False
    retrieval_hit_at_5: bool = False

    # Keyword metrics
    keyword_coverage: float = 0.0
    keywords_found: list[str] = field(default_factory=list)
    keywords_missing: list[str] = field(default_factory=list)

    # No-answer correctness
    no_answer_correctness: bool = False

    # Stability
    latency_ms: int = 0
    error: str | None = None


# ── API mode: call running server ─────────────────────────────────────


async def run_case_api(
    case: EvalCase,
    base_url: str,
    api_key: str,
) -> EvalResult:
    """Run a single eval case against the running API server."""
    import httpx

    result = EvalResult(
        case_id=case.case_id,
        tenant_id=case.tenant_id,
        answer_type=case.answer_type,
    )

    payload = {
        "message": case.question,
        "history": case.history,
    }

    started = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{base_url}/api/v1/assistant/respond",
                json=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        result.error = f"{exc.__class__.__name__}: {exc}"
        result.latency_ms = int((time.perf_counter() - started) * 1000)
        return result

    result.latency_ms = int((time.perf_counter() - started) * 1000)

    # Extract answer
    message = data.get("message", "")
    citations = data.get("citations", [])

    result.answer_text = message
    result.has_answer = bool(message and len(message) > 20)
    result.has_citations = len(citations) > 0
    result.citation_count = len(citations)

    # Extract citation IDs
    result.citation_document_ids = [
        c["document_id"] for c in citations if c.get("document_id")
    ]
    result.citation_source_document_ids = [
        c["source_document_id"] for c in citations if c.get("source_document_id")
    ]

    # Compute metrics
    _compute_metrics(case, result)

    return result


# ── Direct mode: in-process execution ─────────────────────────────────


async def run_case_direct(case: EvalCase) -> EvalResult:
    """Run a single eval case in-process via AssistantService."""

    result = EvalResult(
        case_id=case.case_id,
        tenant_id=case.tenant_id,
        answer_type=case.answer_type,
    )

    try:
        from app.schemas.assistant import AssistantRespondRequest
        from app.services.assistant_service import AssistantService

        svc = AssistantService()
        request = AssistantRespondRequest(
            message=case.question,
            history=case.history,
        )

        started = time.perf_counter()
        response = await svc.respond(
            request=request,
            tenant_id=case.tenant_id,
            user_id=0,  # eval user
            trace_id=f"eval-{case.case_id}",
        )
        result.latency_ms = int((time.perf_counter() - started) * 1000)

        result.answer_text = response.message
        result.has_answer = bool(response.message and len(response.message) > 20)
        result.has_citations = len(response.citations) > 0
        result.citation_count = len(response.citations)

        result.citation_document_ids = [
            c.document_id for c in response.citations if c.document_id
        ]
        result.citation_source_document_ids = [
            c.source_document_id for c in response.citations if c.source_document_id
        ]

    except Exception as exc:
        result.error = f"{exc.__class__.__name__}: {exc}"
        return result

    _compute_metrics(case, result)
    return result


# ── Metric computation ────────────────────────────────────────────────


def _compute_metrics(case: EvalCase, result: EvalResult) -> None:
    """Compute all metrics for a completed eval case."""

    expected_ids = set(case.expected_source_document_ids)

    # ── Citation hit / precision / recall ──
    if expected_ids:
        actual_ids = set(result.citation_source_document_ids) | set(result.citation_document_ids)
        intersection = expected_ids & actual_ids

        result.citation_hit = len(intersection) > 0
        result.citation_precision = (
            len(intersection) / len(actual_ids) if actual_ids else 0.0
        )
        result.citation_recall = (
            len(intersection) / len(expected_ids) if expected_ids else 0.0
        )

    # ── Retrieval hit at K ──
    if expected_ids and result.citation_document_ids:
        cids = result.citation_document_ids
        sids = result.citation_source_document_ids
        all_ids = cids + sids

        result.retrieval_hit_at_1 = bool(expected_ids & set(all_ids[:1]))
        result.retrieval_hit_at_3 = bool(expected_ids & set(all_ids[:3]))
        result.retrieval_hit_at_5 = bool(expected_ids & set(all_ids[:5]))

    # ── Keyword coverage ──
    if case.expected_keywords and result.answer_text:
        answer_lower = result.answer_text.lower()
        found = [k for k in case.expected_keywords if k.lower() in answer_lower]
        missing = [k for k in case.expected_keywords if k.lower() not in answer_lower]

        result.keywords_found = found
        result.keywords_missing = missing
        result.keyword_coverage = (
            len(found) / len(case.expected_keywords) if case.expected_keywords else 0.0
        )

    # ── No-answer correctness ──
    if case.answer_type == "no_answer":
        # Good: system says "could not find" or has no citations
        no_answer_indicators = [
            "could not find",
            "không tìm thấy",
            "không có thông tin",
            "no relevant",
            "không thể trả lời",
        ]
        answer_lower = (result.answer_text or "").lower()
        result.no_answer_correctness = (
            any(ind in answer_lower for ind in no_answer_indicators)
            or not result.has_citations
        )


# ── Report generation ─────────────────────────────────────────────────


def generate_report(results: list[EvalResult]) -> dict:
    """Generate aggregate metrics from individual eval results."""
    total = len(results)
    if total == 0:
        return {"total": 0, "error": "no results"}

    # Group by answer_type
    by_type: dict[str, list[EvalResult]] = {}
    for r in results:
        by_type.setdefault(r.answer_type, []).append(r)

    def _rate(lst: list, attr: str) -> float:
        vals = [getattr(r, attr) for r in lst]
        return round(sum(1 for v in vals if v) / len(vals), 4) if vals else 0.0

    def _avg(lst: list, attr: str) -> float:
        vals = [getattr(r, attr) for r in lst]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    # Aggregate
    report = {
        "total_cases": total,
        "errors": sum(1 for r in results if r.error),
        "aggregate": {
            "has_answer_rate": _rate(results, "has_answer"),
            "has_citations_rate": _rate(results, "has_citations"),
            "avg_citation_count": _avg(results, "citation_count"),
            "avg_keyword_coverage": _avg(results, "keyword_coverage"),
            "avg_latency_ms": _avg(results, "latency_ms"),
        },
        "by_answer_type": {},
        "individual_results": [asdict(r) for r in results],
    }

    # Cases with expected_source_document_ids
    with_expected = [r for r in results if r.citation_hit or r.citation_recall > 0 or any(
        c.expected_source_document_ids for c in [] # placeholder
    )]

    for atype, group in sorted(by_type.items()):
        report["by_answer_type"][atype] = {
            "count": len(group),
            "has_answer_rate": _rate(group, "has_answer"),
            "has_citations_rate": _rate(group, "has_citations"),
            "avg_keyword_coverage": _avg(group, "keyword_coverage"),
            "avg_latency_ms": _avg(group, "latency_ms"),
        }
        if atype == "no_answer":
            report["by_answer_type"][atype]["no_answer_correctness_rate"] = _rate(
                group, "no_answer_correctness"
            )

    return report


# ── Main ──────────────────────────────────────────────────────────────


def load_golden_set(path: str, tenant_filter: str | None = None) -> list[EvalCase]:
    """Load eval cases from JSONL file."""
    cases: list[EvalCase] = []

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                case = EvalCase(
                    case_id=data["case_id"],
                    tenant_id=data["tenant_id"],
                    question=data["question"],
                    history=data.get("history", []),
                    expected_source_document_ids=data.get("expected_source_document_ids", []),
                    expected_keywords=data.get("expected_keywords", []),
                    answer_type=data.get("answer_type") or data.get("expected_answer_type", "general"),
                    notes=data.get("notes", ""),
                )
                if tenant_filter and case.tenant_id != tenant_filter:
                    continue
                cases.append(case)
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning("Skipping line %d: %s", line_num, exc)

    return cases


async def run_eval(args: argparse.Namespace) -> None:
    """Main eval runner."""
    golden_path = args.golden
    if not Path(golden_path).exists():
        logger.error("Golden set not found: %s", golden_path)
        sys.exit(1)

    cases = load_golden_set(golden_path, tenant_filter=args.tenant)
    logger.info("Loaded %d eval cases from %s", len(cases), golden_path)

    if not cases:
        logger.warning("No cases to evaluate")
        return

    results: list[EvalResult] = []

    for i, case in enumerate(cases, start=1):
        logger.info(
            "[%d/%d] Running case=%s tenant=%s type=%s",
            i, len(cases), case.case_id, case.tenant_id, case.answer_type,
        )

        if args.mode == "api":
            result = await run_case_api(case, args.base_url, args.api_key)
        else:
            result = await run_case_direct(case)

        results.append(result)

        # Log individual result
        status = "OK" if not result.error else f"ERROR: {result.error}"
        logger.info(
            "  → %s | answer=%s citations=%d kw_coverage=%.2f latency=%dms",
            status,
            "yes" if result.has_answer else "no",
            result.citation_count,
            result.keyword_coverage,
            result.latency_ms,
        )

    # Generate report
    report = generate_report(results)

    # Output
    output_path = args.output
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    logger.info("Report written to %s", output_path)

    # Print summary
    agg = report["aggregate"]
    print("\n" + "=" * 60)
    print(f"  PHASE 1 EVAL REPORT — {report['total_cases']} cases")
    print("=" * 60)
    print(f"  Errors:             {report['errors']}")
    print(f"  Has Answer Rate:    {agg['has_answer_rate']:.1%}")
    print(f"  Has Citations Rate: {agg['has_citations_rate']:.1%}")
    print(f"  Avg Citations:      {agg['avg_citation_count']:.1f}")
    print(f"  Avg Keyword Cov:    {agg['avg_keyword_coverage']:.1%}")
    print(f"  Avg Latency:        {agg['avg_latency_ms']:.0f}ms")
    print()

    for atype, metrics in report["by_answer_type"].items():
        print(f"  [{atype}] n={metrics['count']}"
              f"  answer={metrics['has_answer_rate']:.0%}"
              f"  citations={metrics['has_citations_rate']:.0%}"
              f"  kw={metrics['avg_keyword_coverage']:.0%}"
              f"  latency={metrics['avg_latency_ms']:.0f}ms")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Phase 1 Eval Runner")
    parser.add_argument(
        "--golden",
        default="evals/phase1/golden_set.jsonl",
        help="Path to golden set JSONL file",
    )
    parser.add_argument(
        "--output",
        default="evals/phase1/results.json",
        help="Path to output results JSON",
    )
    parser.add_argument(
        "--tenant",
        default=None,
        help="Filter to specific tenant_id",
    )
    parser.add_argument(
        "--mode",
        choices=["api", "direct"],
        default="api",
        help="Execution mode: 'api' (call running server) or 'direct' (in-process)",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL for API mode",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("EVAL_API_KEY", ""),
        help="API key for authentication",
    )

    args = parser.parse_args()
    asyncio.run(run_eval(args))


if __name__ == "__main__":
    main()
