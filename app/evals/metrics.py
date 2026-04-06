"""
Phase 1.2A — Metrics aggregation.

Aggregates per-case CaseResult objects into AggregateMetrics,
sliced by overall / tenant / slice_tag / expected_answer_type.

Design rules:
  - Retrieval metrics and answer metrics are strictly separated.
  - No "magic overall score" — each metric is reported individually.
  - Empty groups produce zero-value AggregateMetrics, never errors.
"""
from __future__ import annotations

from collections import defaultdict

from app.evals.schemas import AggregateMetrics, CaseResult


def aggregate(results: list[CaseResult]) -> AggregateMetrics:
    """Compute aggregate metrics for a list of case results."""
    n = len(results)
    if n == 0:
        return AggregateMetrics(count=0)

    # Retrieval
    hit1 = sum(1 for r in results if r.retrieval.hit_at_1) / n
    hit3 = sum(1 for r in results if r.retrieval.hit_at_3) / n
    hit5 = sum(1 for r in results if r.retrieval.hit_at_5) / n
    recall5 = sum(r.retrieval.recall_at_5 for r in results) / n
    mrr_avg = sum(r.retrieval.mrr for r in results) / n

    # Answer
    has_ans = sum(1 for r in results if r.answer.has_answer) / n
    has_cit = sum(1 for r in results if r.answer.has_citations) / n
    cit_valid = sum(1 for r in results if r.answer.citation_doc_ids_valid) / n
    cit_tenant = sum(1 for r in results if r.answer.citation_same_tenant) / n
    kw_avg = sum(r.answer.keyword_coverage for r in results) / n
    forbidden_rate = sum(1 for r in results if r.answer.forbidden_keyword_violation) / n

    # Abstention accuracy: only meaningful for cases that have an abstention expectation
    abstain_cases = [
        r for r in results
        if r.expected_answer_type in (
            "abstain", "no_answer",
            "overview", "specific", "follow_up",
            "citation_required", "ambiguous",
            "general", "compare",
        )
    ]
    abstain_acc = (
        sum(1 for r in abstain_cases if r.answer.abstention_behavior_correct)
        / len(abstain_cases)
        if abstain_cases else 0.0
    )

    # Stability
    avg_lat = sum(r.latency_ms for r in results) / n
    errs = sum(1 for r in results if r.error)

    # 1.2B: Judge score aggregation (only from cases with judge data)
    faith_scores = [
        r.judge_scores["faithfulness"]["score"]
        for r in results
        if r.judge_scores.get("faithfulness")
        and not r.judge_scores["faithfulness"].get("judge_failed")
        and "score" in r.judge_scores["faithfulness"]
    ]
    rel_scores = [
        r.judge_scores["answer_relevance"]["score"]
        for r in results
        if r.judge_scores.get("answer_relevance")
        and not r.judge_scores["answer_relevance"].get("judge_failed")
        and "score" in r.judge_scores["answer_relevance"]
    ]
    judge_fails = sum(
        1 for r in results
        if any(
            isinstance(v, dict) and v.get("judge_failed")
            for v in r.judge_scores.values()
        )
    )

    avg_faith = sum(faith_scores) / len(faith_scores) if faith_scores else 0.0
    avg_rel = sum(rel_scores) / len(rel_scores) if rel_scores else 0.0

    return AggregateMetrics(
        count=n,
        retrieval_hit_at_1=round(hit1, 4),
        retrieval_hit_at_3=round(hit3, 4),
        retrieval_hit_at_5=round(hit5, 4),
        retrieval_recall_at_5=round(recall5, 4),
        retrieval_mrr=round(mrr_avg, 4),
        has_answer_rate=round(has_ans, 4),
        has_citations_rate=round(has_cit, 4),
        citation_doc_ids_valid_rate=round(cit_valid, 4),
        citation_same_tenant_rate=round(cit_tenant, 4),
        keyword_coverage_avg=round(kw_avg, 4),
        forbidden_keyword_violation_rate=round(forbidden_rate, 4),
        abstention_behavior_accuracy=round(abstain_acc, 4),
        avg_latency_ms=round(avg_lat, 2),
        error_count=errs,
        avg_faithfulness=round(avg_faith, 4),
        avg_answer_relevance=round(avg_rel, 4),
        judge_failures=judge_fails,
    )


def aggregate_by_group(
    results: list[CaseResult],
    key_fn,
) -> dict[str, AggregateMetrics]:
    """Aggregate metrics grouped by an arbitrary key function.

    Args:
        results: List of case results.
        key_fn: Function CaseResult -> str (group key).

    Returns:
        Dict mapping group key -> AggregateMetrics.
    """
    groups: dict[str, list[CaseResult]] = defaultdict(list)
    for r in results:
        groups[key_fn(r)].append(r)

    return {k: aggregate(v) for k, v in sorted(groups.items())}


def aggregate_by_slice_tags(
    results: list[CaseResult],
) -> dict[str, AggregateMetrics]:
    """Aggregate grouping by slice_tags.

    A case with multiple slice_tags appears in multiple groups.
    """
    groups: dict[str, list[CaseResult]] = defaultdict(list)
    for r in results:
        tags = r.slice_tags if r.slice_tags else ["_untagged"]
        for tag in tags:
            groups[tag].append(r)

    return {k: aggregate(v) for k, v in sorted(groups.items())}


def compute_all_aggregates(results: list[CaseResult]) -> dict:
    """Compute all aggregate breakdowns in one call.

    Returns dict with keys: overall, by_tenant, by_slice,
    by_expected_answer_type.
    """
    return {
        "overall": aggregate(results),
        "by_tenant": aggregate_by_group(
            results, lambda r: r.tenant_id,
        ),
        "by_slice": aggregate_by_slice_tags(results),
        "by_expected_answer_type": aggregate_by_group(
            results, lambda r: r.expected_answer_type,
        ),
    }
