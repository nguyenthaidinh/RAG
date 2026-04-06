"""
Phase 1.2B — Eval CLI with LLM judges + pairwise comparison.

This script extends 1.2A with:
  - --with-judges flag for LLM judge evaluation
  - --mode pairwise for baseline vs candidate comparison
  - --export-audit-sample for human review export

Usage examples:
    # Standard eval with judges (token-conscious)
    python scripts/run_phase12b_eval.py --mode assistant --tenant demo --with-judges --limit 10

    # Judges on specific slice
    python scripts/run_phase12b_eval.py --mode assistant --tenant demo --with-judges --slice follow_up

    # Custom judge model
    python scripts/run_phase12b_eval.py --mode assistant --tenant demo --with-judges --judge-model gpt-4o

    # Pairwise regression comparison
    python scripts/run_phase12b_eval.py --mode pairwise --baseline path/baseline.jsonl --candidate path/candidate.jsonl

    # Export audit sample (failed cases)
    python scripts/run_phase12b_eval.py --mode assistant --tenant demo --export-audit-sample --audit-strategy failed --limit 10

    # Judges only on failed cases
    python scripts/run_phase12b_eval.py --mode assistant --tenant demo --with-judges --judge-only-failures --limit 10

This script does NOT modify any existing query/assistant flows.
1.2A deterministic graders always run first. Judges are supplementary.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Safe defaults for environment variables
os.environ.setdefault("DATABASE_URL", os.environ.get("DATABASE_URL", ""))
os.environ.setdefault("JWT_SECRET", os.environ.get("JWT_SECRET", "eval-runner"))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("eval_runner_12b")
logger.setLevel(logging.INFO)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 1.2B Eval Runner — LLM judges + pairwise comparison",
    )

    # Dataset
    parser.add_argument(
        "--dataset",
        default="evals/phase1/golden_set.jsonl",
        help="Path to golden set JSONL file",
    )

    # Mode
    parser.add_argument(
        "--mode",
        choices=["assistant", "query", "mixed", "pairwise"],
        default="assistant",
        help="Endpoint mode or pairwise comparison mode",
    )

    # Filters
    parser.add_argument("--tenant", default=None, help="Filter to specific tenant_id")
    parser.add_argument("--case-id", default=None, help="Run a single case by case_id")
    parser.add_argument("--slice", default=None, help="Filter by slice_tag")
    parser.add_argument("--limit", type=int, default=None, help="Max number of cases")

    # Execution
    parser.add_argument(
        "--exec-mode",
        choices=["api", "direct"],
        default="api",
        help="Execution mode: 'api' or 'direct' (in-process)",
    )
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("EVAL_API_KEY", ""),
    )

    # Output
    parser.add_argument(
        "--output-dir",
        default="evals/outputs/phase1",
        help="Directory for output reports",
    )

    # 1.2B: Judge flags
    parser.add_argument(
        "--with-judges",
        action="store_true",
        help="Enable LLM judge grading (costs tokens)",
    )
    parser.add_argument(
        "--judge-model",
        default="gpt-4o-mini",
        help="Model to use for LLM judges",
    )
    parser.add_argument(
        "--judge-only-failures",
        action="store_true",
        help="Only run judges on cases that failed deterministic checks",
    )

    # 1.2B: Pairwise flags
    parser.add_argument(
        "--baseline",
        default=None,
        help="Path to baseline per_case_results.jsonl (pairwise mode)",
    )
    parser.add_argument(
        "--candidate",
        default=None,
        help="Path to candidate per_case_results.jsonl (pairwise mode)",
    )

    # 1.2B: Human audit
    parser.add_argument(
        "--export-audit-sample",
        action="store_true",
        help="Export human audit sample after eval",
    )
    parser.add_argument(
        "--audit-strategy",
        choices=["all", "random", "failed", "high_criticality", "disagreement"],
        default="failed",
        help="Sampling strategy for audit export",
    )
    parser.add_argument(
        "--audit-sample-size",
        type=int,
        default=None,
        help="Max items in audit sample",
    )

    # 1.2B: Validate
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate the dataset, don't run eval",
    )

    # 1.2B: Write audit template
    parser.add_argument(
        "--write-audit-template",
        action="store_true",
        help="Write the human audit annotation template CSV",
    )

    return parser


async def main_async(args: argparse.Namespace) -> None:
    # ── Validate-only mode ────────────────────────────────────────────
    if args.validate_only:
        from app.evals.dataset import validate_dataset

        result = validate_dataset(args.dataset)
        print(f"\nDataset: {args.dataset}")
        print(f"  Total lines:  {result.total_lines}")
        print(f"  Valid:         {result.valid_lines}")
        print(f"  Errors:        {len(result.errors)}")

        if result.errors:
            print("\n  ERRORS:")
            for err in result.errors:
                print(f"    Line {err['line']}: {err['error']}")
            sys.exit(1)
        else:
            print("  ✓ All cases valid")
            sys.exit(0)

    # ── Write audit template ──────────────────────────────────────────
    if args.write_audit_template:
        from app.evals.audit import write_audit_template

        path = write_audit_template()
        print(f"✓ Audit template written: {path}")
        sys.exit(0)

    # ── Pairwise mode ─────────────────────────────────────────────────
    if args.mode == "pairwise":
        await _run_pairwise(args)
        return

    # ── Standard eval mode (with optional judges) ─────────────────────
    await _run_standard(args)


async def _run_standard(args: argparse.Namespace) -> None:
    """Standard eval with optional LLM judges."""
    from app.evals.reporting import (
        print_summary,
        write_judge_reports,
        write_reports,
    )
    from app.evals.runner import RunConfig, run_eval

    config = RunConfig(
        dataset_path=args.dataset,
        mode=args.mode,
        tenant=args.tenant,
        case_id=args.case_id,
        slice_tag=args.slice,
        limit=args.limit,
        output_dir=args.output_dir,
        exec_mode=args.exec_mode,
        base_url=args.base_url,
        api_key=args.api_key,
    )

    # Run 1.2A deterministic eval
    run_result = await run_eval(config)

    if not run_result.results:
        logger.warning("No results to report")
        return

    # 1.2B: Run judges (if enabled)
    if args.with_judges:
        await _run_judges(
            run_result.results,
            judge_model=args.judge_model,
            judge_only_failures=args.judge_only_failures,
        )

    # Write 1.2A reports
    paths = write_reports(
        run_result.results,
        mode=config.mode,
        tenant=config.tenant,
        output_dir=config.output_dir,
    )

    # Write 1.2B judge reports
    if args.with_judges:
        judge_paths = write_judge_reports(
            run_result.results,
            mode=config.mode,
            tenant=config.tenant,
            judge_model=args.judge_model,
            output_dir=config.output_dir,
        )
        paths.update(judge_paths)

    # Export audit sample
    if args.export_audit_sample:
        from app.evals.audit import export_audit_sample

        # Build judge_results dict for audit
        judge_results_map = {}
        for r in run_result.results:
            if r.judge_scores:
                judge_results_map[r.case_id] = r.judge_scores

        audit_path = export_audit_sample(
            [r.model_dump() for r in run_result.results],
            judge_results=judge_results_map,
            strategy=args.audit_strategy,
            sample_size=args.audit_sample_size,
            output_dir=config.output_dir,
        )
        paths["human_audit_sample"] = audit_path

    for name, path in paths.items():
        logger.info("Report written: %s -> %s", name, path)

    # Print summary
    print_summary(
        run_result.results, mode=config.mode, tenant=config.tenant,
    )

    logger.info("Total eval time: %dms", run_result.elapsed_ms)


async def _run_judges(
    results: list,
    *,
    judge_model: str = "gpt-4o-mini",
    judge_only_failures: bool = False,
) -> None:
    """Apply LLM judges to case results (in-place mutation)."""
    from app.evals.graders.llm_judge import (
        JudgeConfig,
        run_judges_for_case,
    )

    cfg = JudgeConfig(model=judge_model)

    candidates = results
    if judge_only_failures:
        candidates = [
            r for r in results
            if r.error
            or not r.answer.has_answer
            or r.answer.forbidden_keyword_violation
            or not r.answer.abstention_behavior_correct
            or r.answer.keyword_coverage < 0.5
        ]
        logger.info(
            "Judge-only-failures: %d / %d cases selected",
            len(candidates), len(results),
        )

    for i, r in enumerate(candidates, start=1):
        logger.info(
            "[judge %d/%d] case=%s model=%s",
            i, len(candidates), r.case_id, judge_model,
        )

        try:
            judge_results = await run_judges_for_case(
                case_id=r.case_id,
                question=r.question,
                answer_text=r.answer_text,
                citations=None,  # Uses the data from case result
                expected_answer_type=r.expected_answer_type,
                config=cfg,
            )

            # Store as serializable dict
            r.judge_scores = {
                jtype: asdict(jresult)
                for jtype, jresult in judge_results.items()
            }

            # Log summary
            for jtype, jresult in judge_results.items():
                if jresult.judge_failed:
                    logger.warning(
                        "  → %s FAILED: %s", jtype, jresult.error,
                    )
                else:
                    logger.info(
                        "  → %s score=%d verdict=%s lat=%dms%s",
                        jtype, jresult.score, jresult.verdict,
                        jresult.latency_ms,
                        " (cached)" if jresult.cached else "",
                    )

        except Exception as exc:
            logger.error("Judge error for %s: %s", r.case_id, exc)
            r.judge_scores = {
                "faithfulness": {
                    "judge_type": "faithfulness",
                    "judge_failed": True,
                    "error": str(exc),
                },
                "answer_relevance": {
                    "judge_type": "answer_relevance",
                    "judge_failed": True,
                    "error": str(exc),
                },
            }


async def _run_pairwise(args: argparse.Namespace) -> None:
    """Run pairwise regression comparison."""
    from app.evals.pairwise import (
        align_results,
        load_per_case_results,
        pairwise_judge,
    )
    from app.evals.reporting import (
        print_pairwise_summary,
        write_pairwise_reports,
    )

    if not args.baseline or not args.candidate:
        print("ERROR: --baseline and --candidate paths are required for pairwise mode")
        sys.exit(1)

    logger.info("Loading baseline: %s", args.baseline)
    baseline = load_per_case_results(args.baseline)

    logger.info("Loading candidate: %s", args.candidate)
    candidate = load_per_case_results(args.candidate)

    pairs = align_results(baseline, candidate)

    if not pairs:
        print("ERROR: No matching cases found between baseline and candidate")
        sys.exit(1)

    # Apply filters
    if args.tenant:
        pairs = [(a, b) for a, b in pairs if a.get("tenant_id") == args.tenant]

    if args.slice:
        pairs = [
            (a, b) for a, b in pairs
            if args.slice in a.get("slice_tags", [])
        ]

    if args.limit:
        pairs = pairs[:args.limit]

    logger.info("Running pairwise comparison on %d cases", len(pairs))

    pairwise_results = []
    for i, (a, b) in enumerate(pairs, start=1):
        case_id = a.get("case_id", "?")
        logger.info("[pairwise %d/%d] case=%s", i, len(pairs), case_id)

        result = await pairwise_judge(
            case_id=case_id,
            tenant_id=a.get("tenant_id", ""),
            question=a.get("question", ""),
            answer_a=a.get("answer_text", "") or "",
            answer_b=b.get("answer_text", "") or "",
            expected_answer_type=a.get("expected_answer_type", "specific"),
            slice_tags=a.get("slice_tags", []),
            model=args.judge_model,
        )

        pairwise_results.append(result)

        status = "OK" if not result.judge_failed else f"ERR: {result.error}"
        logger.info(
            "  → %s winner=%s lat=%dms", status, result.winner, result.latency_ms,
        )

    # Write reports
    paths = write_pairwise_reports(
        pairwise_results, output_dir=args.output_dir,
    )

    for name, path in paths.items():
        logger.info("Report written: %s -> %s", name, path)

    # Print summary
    print_pairwise_summary(pairwise_results)


def main():
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
