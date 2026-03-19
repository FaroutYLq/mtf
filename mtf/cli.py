"""CLI entry point for the mtf command."""

from __future__ import annotations

import argparse
import asyncio
import sys

from mtf.config import MTFConfig
from mtf.orchestrator import MTFOrchestrator


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mtf",
        description="My Theorist Friend — multi-agent physics research assistant",
    )
    p.add_argument(
        "phenomenon",
        nargs="?",
        help="Description of the experimental phenomenon (or omit to be prompted)",
    )
    p.add_argument("--n-literature", type=int, default=3)
    p.add_argument("--n-fitting", type=int, default=3)
    p.add_argument("--n-reviewer", type=int, default=3)
    p.add_argument("--literature-model", default="claude-opus-4-6")
    p.add_argument("--fitting-model", default="claude-opus-4-6")
    p.add_argument("--reviewer-model", default="claude-opus-4-6")
    p.add_argument("--debate-model", default="claude-opus-4-6")
    p.add_argument("--max-debate-rounds", type=int, default=3)
    p.add_argument(
        "--images",
        nargs="+",
        metavar="PATH",
        default=[],
        help="Image files (plots, figures) to digest before analysis",
    )
    p.add_argument("--image-digest-model", default="claude-opus-4-6")
    return p


def main() -> None:
    args = build_parser().parse_args()
    phenomenon = args.phenomenon
    if not phenomenon:
        try:
            phenomenon = input("Describe the experimental phenomenon:\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            sys.exit(0)
    if not phenomenon:
        print("No phenomenon provided. Exiting.")
        sys.exit(1)

    config = MTFConfig(
        n_literature=args.n_literature,
        n_fitting=args.n_fitting,
        n_reviewer=args.n_reviewer,
        literature_model=args.literature_model,
        fitting_model=args.fitting_model,
        reviewer_model=args.reviewer_model,
        debate_model=args.debate_model,
        max_debate_rounds=args.max_debate_rounds,
        image_digest_model=args.image_digest_model,
    )
    orchestrator = MTFOrchestrator(config=config)
    asyncio.run(orchestrator.run(phenomenon, images=args.images or None))
