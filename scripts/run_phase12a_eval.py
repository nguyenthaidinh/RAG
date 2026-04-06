"""
Phase 1.2A — Eval CLI entry point.

Usage examples:
    python scripts/run_phase12a_eval.py --mode assistant --tenant demo
    python scripts/run_phase12a_eval.py --mode query --tenant demo
    python scripts/run_phase12a_eval.py --case-id demo_overview_001
    python scripts/run_phase12a_eval.py --slice overview
    python scripts/run_phase12a_eval.py --limit 5
    python scripts/run_phase12a_eval.py --dataset evals/phase1/golden_set.jsonl
    python scripts/run_phase12a_eval.py --exec-mode direct --tenant demo

This script does NOT modify any existing query/assistant flows.
It is purely an evaluation harness.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Safe defaults for environment variables (eval doesn't need real DB)
os.environ.setdefault("DATABASE_URL", os.environ.get("DATABASE_URL", ""))
os.environ.setdefault("JWT_SECRET", os.environ.get("JWT_SECRET", "eval-runner"))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("eval_runner_12a")
logger.setLevel(logging.INFO)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 1.2A Eval Runner — tenant-aware, deterministic evaluation",
    )

    # Dataset
    parser.add_argument(
        "--dataset",
        default="evals/phase1/golden_set.jsonl",
        help="Path to golden set JSONL file",
    )

    # Filters
    parser.add_argument(
        "--mode",
        choices=["assistant", "query", "mixed"],
        default="assistant",
        help="Endpoint mode filter: assistant, query, or mixed (both)",
    )
    parser.add_argument(
        "--tenant",
        default=None,
        help="Filter to specific tenant_id",
    )
    parser.add_argument(
        "--case-id",
        default=None,
        help="Run a single case by case_id",
    )
    parser.add_argument(
        "--slice",
        default=None,
        help="Filter by slice_tag",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of cases to evaluate",
    )

    # Execution
    parser.add_argument(
        "--exec-mode",
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

    # Output
    parser.add_argument(
        "--output-dir",
        default="evals/outputs/phase1",
        help="Directory for output reports",
    )

    # Flags
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate the dataset, don't run eval",
    )

    return parser


async def main_async(args: argparse.Namespace) -> None:
    from app.evals.dataset import validate_dataset
    from app.evals.reporting import print_summary, write_reports
    from app.evals.runner import RunConfig, run_eval

    # ── Validate-only mode ────────────────────────────────────────────
    if args.validate_only:
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

    # ── Build config ──────────────────────────────────────────────────
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

    # ── Run ───────────────────────────────────────────────────────────
    run_result = await run_eval(config)

    if not run_result.results:
        logger.warning("No results to report")
        return

    # ── Write reports ─────────────────────────────────────────────────
    paths = write_reports(
        run_result.results,
        mode=config.mode,
        tenant=config.tenant,
        output_dir=config.output_dir,
    )

    for name, path in paths.items():
        logger.info("Report written: %s -> %s", name, path)

    # ── Print summary ─────────────────────────────────────────────────
    print_summary(
        run_result.results,
        mode=config.mode,
        tenant=config.tenant,
    )

    logger.info("Total eval time: %dms", run_result.elapsed_ms)


def main():
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
