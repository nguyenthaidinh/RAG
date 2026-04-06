"""
Phase 1.2A+B — Report writer.

Writes evaluation outputs to the standard output folder:
  evals/outputs/phase1/
    ├── summary.json          # Run-level aggregate report
    ├── per_case_results.jsonl # One result per line
    ├── retrieval_report.csv   # Retrieval metrics per case
    ├── answer_report.csv      # Answer metrics per case
    ├── judge_report.json      # 1.2B: Judge aggregate report
    ├── judge_report.csv       # 1.2B: Judge per-case scores
    ├── pairwise_report.json   # 1.2B: Pairwise aggregate report
    └── pairwise_report.csv    # 1.2B: Pairwise per-case results

All outputs are human-readable and machine-parseable.
"""
from __future__ import annotations

import csv
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path

from app.evals.metrics import compute_all_aggregates
from app.evals.schemas import CaseResult, RunSummary

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = "evals/outputs/phase1"


def _get_git_commit() -> str | None:
    """Try to get current git commit hash. Returns None on failure."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def write_reports(
    results: list[CaseResult],
    *,
    mode: str,
    tenant: str | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, str]:
    """Write all report files and return paths.

    Args:
        results: List of evaluated CaseResult objects.
        mode: "assistant" | "query" | "mixed".
        tenant: Tenant filter used for this run (or None).
        output_dir: Directory to write reports to.

    Returns:
        Dict mapping report name -> file path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Timestamp for file naming (avoid overwriting)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")

    paths: dict[str, str] = {}

    # ── 1. Per-case results (JSONL) ───────────────────────────────────
    per_case_path = output_dir / "per_case_results.jsonl"
    with open(per_case_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r.model_dump(), ensure_ascii=False, default=str) + "\n")
    paths["per_case_results"] = str(per_case_path)

    # ── 2. Summary JSON ──────────────────────────────────────────────
    aggregates = compute_all_aggregates(results)

    evaluated = [r for r in results if not r.skipped]
    skipped = [r for r in results if r.skipped]

    summary = RunSummary(
        timestamp=datetime.utcnow().isoformat(),
        git_commit=_get_git_commit(),
        mode=mode,
        tenant=tenant,
        evaluated_cases=len(evaluated),
        skipped_cases=len(skipped),
        overall=aggregates["overall"],
        by_tenant=aggregates["by_tenant"],
        by_slice=aggregates["by_slice"],
        by_expected_answer_type=aggregates["by_expected_answer_type"],
    )

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary.model_dump(), f, indent=2, ensure_ascii=False, default=str)
    paths["summary"] = str(summary_path)

    # ── 3. Retrieval report (CSV) ─────────────────────────────────────
    retrieval_path = output_dir / "retrieval_report.csv"
    _write_retrieval_csv(results, retrieval_path)
    paths["retrieval_report"] = str(retrieval_path)

    # ── 4. Answer report (CSV) ────────────────────────────────────────
    answer_path = output_dir / "answer_report.csv"
    _write_answer_csv(results, answer_path)
    paths["answer_report"] = str(answer_path)

    return paths


def _write_retrieval_csv(results: list[CaseResult], path: Path) -> None:
    """Write retrieval metrics per case to CSV."""
    fieldnames = [
        "case_id",
        "tenant_id",
        "mode",
        "question",
        "expected_source_doc_ids",
        "retrieved_doc_ids",
        "hit_at_1",
        "hit_at_3",
        "hit_at_5",
        "recall_at_5",
        "mrr",
        "criticality",
        "error",
    ]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "case_id": r.case_id,
                "tenant_id": r.tenant_id,
                "mode": r.mode,
                "question": r.question[:100],
                "expected_source_doc_ids": json.dumps(r.expected_source_document_ids),
                "retrieved_doc_ids": json.dumps(r.retrieved_document_ids),
                "hit_at_1": r.retrieval.hit_at_1,
                "hit_at_3": r.retrieval.hit_at_3,
                "hit_at_5": r.retrieval.hit_at_5,
                "recall_at_5": round(r.retrieval.recall_at_5, 4),
                "mrr": round(r.retrieval.mrr, 4),
                "criticality": r.criticality,
                "error": r.error or "",
            })


def _write_answer_csv(results: list[CaseResult], path: Path) -> None:
    """Write answer metrics per case to CSV."""
    fieldnames = [
        "case_id",
        "tenant_id",
        "mode",
        "question",
        "expected_answer_type",
        "has_answer",
        "has_citations",
        "citation_doc_ids_valid",
        "citation_same_tenant",
        "keyword_coverage",
        "forbidden_violation",
        "abstention_detected",
        "abstention_correct",
        "answer_preview",
        "criticality",
        "error",
    ]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            preview = (r.answer_text or "")[:120].replace("\n", " ")
            writer.writerow({
                "case_id": r.case_id,
                "tenant_id": r.tenant_id,
                "mode": r.mode,
                "question": r.question[:100],
                "expected_answer_type": r.expected_answer_type,
                "has_answer": r.answer.has_answer,
                "has_citations": r.answer.has_citations,
                "citation_doc_ids_valid": r.answer.citation_doc_ids_valid,
                "citation_same_tenant": r.answer.citation_same_tenant,
                "keyword_coverage": round(r.answer.keyword_coverage, 4),
                "forbidden_violation": r.answer.forbidden_keyword_violation,
                "abstention_detected": r.answer.abstention_detected,
                "abstention_correct": r.answer.abstention_behavior_correct,
                "answer_preview": preview,
                "criticality": r.criticality,
                "error": r.error or "",
            })


def print_summary(results: list[CaseResult], *, mode: str, tenant: str | None) -> None:
    """Print a human-readable summary to stdout."""
    aggregates = compute_all_aggregates(results)
    overall = aggregates["overall"]

    evaluated = [r for r in results if not r.skipped]
    skipped = [r for r in results if r.skipped]

    print()
    print("=" * 70)
    print(f"  PHASE 1.2A EVAL REPORT — {len(evaluated)} cases evaluated")
    print(f"  Mode: {mode}  |  Tenant: {tenant or 'all'}")
    print("=" * 70)
    print()

    print("  RETRIEVAL METRICS:")
    print(f"    Hit@1:      {overall.retrieval_hit_at_1:.1%}")
    print(f"    Hit@3:      {overall.retrieval_hit_at_3:.1%}")
    print(f"    Hit@5:      {overall.retrieval_hit_at_5:.1%}")
    print(f"    Recall@5:   {overall.retrieval_recall_at_5:.1%}")
    print(f"    MRR:        {overall.retrieval_mrr:.4f}")
    print()

    print("  ANSWER METRICS:")
    print(f"    Has Answer Rate:              {overall.has_answer_rate:.1%}")
    print(f"    Has Citations Rate:           {overall.has_citations_rate:.1%}")
    print(f"    Citation Doc IDs Valid Rate:   {overall.citation_doc_ids_valid_rate:.1%}")
    print(f"    Citation Same Tenant Rate:     {overall.citation_same_tenant_rate:.1%}")
    print(f"    Keyword Coverage Avg:          {overall.keyword_coverage_avg:.1%}")
    print(f"    Forbidden Keyword Violation:   {overall.forbidden_keyword_violation_rate:.1%}")
    print(f"    Abstention Behavior Accuracy:  {overall.abstention_behavior_accuracy:.1%}")
    print()

    print("  STABILITY:")
    print(f"    Avg Latency:  {overall.avg_latency_ms:.0f}ms")
    print(f"    Errors:       {overall.error_count}")
    print(f"    Skipped:      {len(skipped)}")
    print()

    # By expected_answer_type
    by_type = aggregates["by_expected_answer_type"]
    if by_type:
        print("  BY ANSWER TYPE:")
        for atype, metrics in by_type.items():
            print(
                f"    [{atype}] n={metrics.count}"
                f"  answer={metrics.has_answer_rate:.0%}"
                f"  citations={metrics.has_citations_rate:.0%}"
                f"  kw={metrics.keyword_coverage_avg:.0%}"
                f"  hit@3={metrics.retrieval_hit_at_3:.0%}"
            )
        print()

    # By tenant
    by_tenant = aggregates["by_tenant"]
    if by_tenant and len(by_tenant) > 1:
        print("  BY TENANT:")
        for tid, metrics in by_tenant.items():
            print(
                f"    [{tid}] n={metrics.count}"
                f"  answer={metrics.has_answer_rate:.0%}"
                f"  citations={metrics.has_citations_rate:.0%}"
            )
        print()

    # 1.2B: Judge metrics (only if judge data is present)
    if overall.avg_faithfulness > 0 or overall.avg_answer_relevance > 0:
        print("  LLM JUDGE METRICS:")
        print(f"    Avg Faithfulness (0-5):       {overall.avg_faithfulness:.2f}")
        print(f"    Avg Answer Relevance (0-5):   {overall.avg_answer_relevance:.2f}")
        print(f"    Judge Failures:               {overall.judge_failures}")
        print()

    print("=" * 70)


# ── 1.2B: Judge reports ───────────────────────────────────────────────────────


def write_judge_reports(
    results: list[CaseResult],
    *,
    mode: str,
    tenant: str | None = None,
    judge_model: str | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, str]:
    """Write judge-specific report files. Returns paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, str] = {}

    # Cases that have judge scores
    judged = [r for r in results if r.judge_scores]
    if not judged:
        return paths

    # Judge report JSON (aggregate)
    judge_summary = {
        "timestamp": datetime.utcnow().isoformat(),
        "mode": mode,
        "tenant": tenant,
        "judge_model": judge_model,
        "judged_cases": len(judged),
        "judge_failures": sum(
            1 for r in judged
            if any(
                isinstance(v, dict) and v.get("judge_failed")
                for v in r.judge_scores.values()
            )
        ),
    }

    # Per-judge-type aggregates
    for judge_type in ["faithfulness", "answer_relevance"]:
        scores = [
            r.judge_scores[judge_type]["score"]
            for r in judged
            if judge_type in r.judge_scores
            and not r.judge_scores[judge_type].get("judge_failed")
            and "score" in r.judge_scores[judge_type]
        ]
        judge_summary[f"avg_{judge_type}"] = round(
            sum(scores) / len(scores), 4,
        ) if scores else 0.0
        judge_summary[f"{judge_type}_count"] = len(scores)

    judge_json_path = output_dir / "judge_report.json"
    with open(judge_json_path, "w", encoding="utf-8") as f:
        json.dump(judge_summary, f, indent=2, ensure_ascii=False, default=str)
    paths["judge_report_json"] = str(judge_json_path)

    # Judge report CSV (per-case)
    judge_csv_path = output_dir / "judge_report.csv"
    fieldnames = [
        "case_id", "tenant_id", "mode",
        "question", "expected_answer_type",
        "faithfulness_score", "faithfulness_verdict",
        "faithfulness_rationale", "faithfulness_failed",
        "answer_relevance_score", "answer_relevance_verdict",
        "answer_relevance_rationale", "answer_relevance_failed",
        "answer_preview", "criticality",
    ]

    with open(judge_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in judged:
            faith = r.judge_scores.get("faithfulness", {})
            rel = r.judge_scores.get("answer_relevance", {})
            writer.writerow({
                "case_id": r.case_id,
                "tenant_id": r.tenant_id,
                "mode": r.mode,
                "question": r.question[:100],
                "expected_answer_type": r.expected_answer_type,
                "faithfulness_score": faith.get("score", ""),
                "faithfulness_verdict": faith.get("verdict", ""),
                "faithfulness_rationale": (faith.get("rationale", "") or "")[:200],
                "faithfulness_failed": faith.get("judge_failed", False),
                "answer_relevance_score": rel.get("score", ""),
                "answer_relevance_verdict": rel.get("verdict", ""),
                "answer_relevance_rationale": (rel.get("rationale", "") or "")[:200],
                "answer_relevance_failed": rel.get("judge_failed", False),
                "answer_preview": (r.answer_text or "")[:120].replace("\n", " "),
                "criticality": r.criticality,
            })

    paths["judge_report_csv"] = str(judge_csv_path)
    return paths


# ── 1.2B: Pairwise reports ────────────────────────────────────────────────────


def write_pairwise_reports(
    pairwise_results: list,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, str]:
    """Write pairwise comparison report files. Returns paths."""
    from app.evals.pairwise import (
        PairwiseResult,
        aggregate_pairwise,
        aggregate_pairwise_by_group,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, str] = {}

    if not pairwise_results:
        return paths

    # Pairwise aggregate JSON
    overall = aggregate_pairwise(pairwise_results)
    by_tenant = aggregate_pairwise_by_group(
        pairwise_results, lambda r: r.tenant_id,
    )
    by_type = aggregate_pairwise_by_group(
        pairwise_results, lambda r: r.expected_answer_type,
    )

    pairwise_summary = {
        "timestamp": datetime.utcnow().isoformat(),
        "overall": overall.to_dict(),
        "by_tenant": {k: v.to_dict() for k, v in by_tenant.items()},
        "by_expected_answer_type": {k: v.to_dict() for k, v in by_type.items()},
    }

    pw_json_path = output_dir / "pairwise_report.json"
    with open(pw_json_path, "w", encoding="utf-8") as f:
        json.dump(pairwise_summary, f, indent=2, ensure_ascii=False, default=str)
    paths["pairwise_report_json"] = str(pw_json_path)

    # Pairwise per-case CSV
    pw_csv_path = output_dir / "pairwise_report.csv"
    fieldnames = [
        "case_id", "tenant_id", "expected_answer_type",
        "winner", "rationale",
        "answer_a_preview", "answer_b_preview",
        "judge_failed", "error", "latency_ms",
    ]

    with open(pw_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in pairwise_results:
            writer.writerow({
                "case_id": r.case_id,
                "tenant_id": r.tenant_id,
                "expected_answer_type": r.expected_answer_type,
                "winner": r.winner,
                "rationale": (r.rationale or "")[:200],
                "answer_a_preview": r.answer_a_preview[:80],
                "answer_b_preview": r.answer_b_preview[:80],
                "judge_failed": r.judge_failed,
                "error": r.error or "",
                "latency_ms": r.latency_ms,
            })

    paths["pairwise_report_csv"] = str(pw_csv_path)
    return paths


def print_pairwise_summary(pairwise_results: list) -> None:
    """Print pairwise comparison summary to stdout."""
    from app.evals.pairwise import aggregate_pairwise, aggregate_pairwise_by_group

    if not pairwise_results:
        return

    overall = aggregate_pairwise(pairwise_results)

    print()
    print("=" * 70)
    print(f"  PAIRWISE COMPARISON — {overall.total} cases compared")
    print("=" * 70)
    print()
    print(f"    Baseline (A) Wins:   {overall.wins_a:3d}  ({overall.win_rate_a:.1%})")
    print(f"    Candidate (B) Wins:  {overall.wins_b:3d}  ({overall.win_rate_b:.1%})")
    print(f"    Ties:                {overall.ties:3d}  ({overall.tie_rate:.1%})")
    print(f"    Judge Failures:      {overall.judge_failures}")
    print()

    by_type = aggregate_pairwise_by_group(
        pairwise_results, lambda r: r.expected_answer_type,
    )
    if by_type:
        print("  BY ANSWER TYPE:")
        for atype, agg in by_type.items():
            print(
                f"    [{atype}] n={agg.total}"
                f"  A={agg.win_rate_a:.0%}"
                f"  B={agg.win_rate_b:.0%}"
                f"  tie={agg.tie_rate:.0%}"
            )
        print()

    print("=" * 70)
