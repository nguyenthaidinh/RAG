"""
Phase 1.2B — Human audit export hook.

Exports evaluation data for human review without building a UI.

Outputs:
  - human_audit_sample.jsonl  — Rich per-case data for review
  - human_audit_template.csv  — Empty template for human annotations

Design rules:
  - No UI. File export only.
  - Each item has full context: question, answer, metrics, judge scores.
  - Sampling strategies: random, failed, high_criticality, disagreement.
  - Template CSV provides the annotation schema.
"""
from __future__ import annotations

import csv
import json
import logging
import random
from dataclasses import asdict
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)


SamplingStrategy = Literal["all", "random", "failed", "high_criticality", "disagreement"]


def export_audit_sample(
    results: list[dict],
    *,
    judge_results: dict[str, dict] | None = None,
    pairwise_results: dict[str, dict] | None = None,
    strategy: SamplingStrategy = "all",
    sample_size: int | None = None,
    output_dir: str | Path = "evals/outputs/phase1",
) -> str:
    """Export human audit sample to JSONL.

    Args:
        results: List of CaseResult dicts (from per_case_results.jsonl
                 or CaseResult.model_dump()).
        judge_results: Optional dict mapping case_id -> {judge_type: JudgeResult dict}.
        pairwise_results: Optional dict mapping case_id -> PairwiseResult dict.
        strategy: Sampling strategy for selecting cases.
        sample_size: Max number of cases in the sample.
        output_dir: Output directory.

    Returns:
        Path to the written audit sample file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Apply sampling strategy
    sampled = _apply_strategy(
        results, judge_results or {}, strategy,
    )

    # Apply limit
    if sample_size is not None and len(sampled) > sample_size:
        if strategy == "random":
            sampled = random.sample(sampled, sample_size)
        else:
            sampled = sampled[:sample_size]

    # Build audit items
    audit_items: list[dict] = []
    for r in sampled:
        case_id = r.get("case_id", "?")

        item: dict = {
            # Identity
            "case_id": case_id,
            "tenant_id": r.get("tenant_id", ""),
            "question": r.get("question", ""),
            "expected_answer_type": r.get("expected_answer_type", ""),
            "expected_source_document_ids": r.get("expected_source_document_ids", []),
            "criticality": r.get("criticality", "medium"),
            "slice_tags": r.get("slice_tags", []),

            # History
            "history": r.get("history", []),

            # Outputs
            "answer_text": r.get("answer_text"),
            "citation_document_ids": r.get("citation_document_ids", []),
            "retrieved_document_ids": r.get("retrieved_document_ids", []),

            # Deterministic metrics
            "retrieval_metrics": r.get("retrieval", {}),
            "answer_metrics": r.get("answer", {}),

            # Error info
            "error": r.get("error"),
            "failure_reasons": r.get("failure_reasons", []),
            "latency_ms": r.get("latency_ms", 0),
        }

        # Judge scores
        if case_id in (judge_results or {}):
            item["judge_scores"] = judge_results[case_id]

        # Pairwise result
        if case_id in (pairwise_results or {}):
            item["pairwise_result"] = pairwise_results[case_id]

        audit_items.append(item)

    # Write JSONL
    path = output_dir / "human_audit_sample.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for item in audit_items:
            f.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")

    logger.info(
        "Audit sample exported: %d items -> %s (strategy=%s)",
        len(audit_items), path, strategy,
    )
    return str(path)


def write_audit_template(
    output_dir: str | Path = "evals/phase1",
) -> str:
    """Write the human audit annotation template CSV.

    Returns:
        Path to the template file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / "human_audit_template.csv"
    fieldnames = [
        "case_id",
        "reviewer",
        "human_faithfulness",      # 0-5
        "human_relevance",         # 0-5
        "human_completeness",      # 0-5
        "human_preferred_variant",  # "A" | "B" | "tie" | ""
        "human_answer_quality",     # "good" | "acceptable" | "poor"
        "comments",
    ]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        # Write one example row
        writer.writerow({
            "case_id": "example_001",
            "reviewer": "reviewer_name",
            "human_faithfulness": "4",
            "human_relevance": "5",
            "human_completeness": "3",
            "human_preferred_variant": "",
            "human_answer_quality": "good",
            "comments": "Example row — delete this and fill in your assessments",
        })

    logger.info("Audit template written: %s", path)
    return str(path)


# ── Sampling strategies ───────────────────────────────────────────────────────


def _apply_strategy(
    results: list[dict],
    judge_results: dict[str, dict],
    strategy: SamplingStrategy,
) -> list[dict]:
    """Apply sampling strategy to select cases for audit."""

    if strategy == "all":
        return list(results)

    if strategy == "random":
        return list(results)  # Random sampling happens after via sample_size

    if strategy == "failed":
        return _filter_failed(results, judge_results)

    if strategy == "high_criticality":
        return [r for r in results if r.get("criticality") == "high"]

    if strategy == "disagreement":
        return _filter_disagreement(results, judge_results)

    return list(results)


def _filter_failed(
    results: list[dict],
    judge_results: dict[str, dict],
) -> list[dict]:
    """Select cases that failed deterministic or judge checks."""
    failed = []
    for r in results:
        case_id = r.get("case_id", "")

        # Deterministic failures
        answer_metrics = r.get("answer", {})
        has_failure = (
            r.get("error")
            or not answer_metrics.get("has_answer", True)
            or answer_metrics.get("forbidden_keyword_violation", False)
            or not answer_metrics.get("abstention_behavior_correct", True)
        )

        # Judge failures
        if case_id in judge_results:
            for jtype, jresult in judge_results[case_id].items():
                if isinstance(jresult, dict):
                    if jresult.get("judge_failed"):
                        has_failure = True
                    elif jresult.get("score", 5) <= 2:
                        has_failure = True

        if has_failure:
            failed.append(r)

    return failed


def _filter_disagreement(
    results: list[dict],
    judge_results: dict[str, dict],
) -> list[dict]:
    """Select cases where deterministic and judge metrics disagree.

    E.g., deterministic says answer is good, but judge gives low score.
    """
    disagreements = []
    for r in results:
        case_id = r.get("case_id", "")
        answer_metrics = r.get("answer", {})
        has_answer = answer_metrics.get("has_answer", False)
        kw_coverage = answer_metrics.get("keyword_coverage", 0)

        if case_id not in judge_results:
            continue

        for jtype, jresult in judge_results[case_id].items():
            if not isinstance(jresult, dict) or jresult.get("judge_failed"):
                continue
            score = jresult.get("score", 3)

            # Deterministic good but judge poor
            if has_answer and kw_coverage > 0.5 and score <= 2:
                disagreements.append(r)
                break

            # Deterministic poor but judge good
            if (not has_answer or kw_coverage < 0.3) and score >= 4:
                disagreements.append(r)
                break

    return disagreements
