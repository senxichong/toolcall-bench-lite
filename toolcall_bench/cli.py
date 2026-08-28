"""Command-line interface for running benchmarks and rendering reports."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

from . import __version__
from .agent import MockAgent, ModelAdapter, OpenAICompatibleAdapter
from .report import report_from_json, write_markdown_report
from .runner import BenchmarkRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="toolcall-bench",
        description="Lightweight evaluation for LLM tool/function calling.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    run_parser = commands.add_parser("run", help="run a JSONL benchmark")
    run_parser.add_argument("benchmark", type=Path, help="path to benchmark JSONL")
    run_parser.add_argument(
        "--adapter",
        choices=("mock", "openai-compatible"),
        default="mock",
        help="model adapter (default: mock)",
    )
    run_parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/latest.json"),
        help="JSON output path (default: runs/latest.json)",
    )
    run_parser.add_argument(
        "--markdown",
        type=Path,
        help="Markdown path (default: JSON output with .md suffix)",
    )
    run_parser.add_argument(
        "--model",
        help="OpenAI-compatible model; defaults to TOOLCALL_BENCH_MODEL",
    )
    run_parser.add_argument(
        "--base-url",
        help="OpenAI-compatible base URL; defaults to TOOLCALL_BENCH_BASE_URL",
    )
    run_parser.add_argument("--timeout", type=float, default=60.0)

    report_parser = commands.add_parser("report", help="render Markdown from run JSON")
    report_parser.add_argument("run_json", type=Path, help="path to run JSON")
    report_parser.add_argument("--output", type=Path, help="Markdown output path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run":
            return _run_command(args)
        return _report_command(args)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _run_command(args: argparse.Namespace) -> int:
    adapter = _build_adapter(args)
    report = BenchmarkRunner(adapter=adapter).run(args.benchmark, args.output)
    markdown_path = args.markdown or args.output.with_suffix(".md")
    write_markdown_report(report, markdown_path)
    summary = report.summary()
    overall = summary["overall_success"]
    print(f"Completed {overall['total']} tasks with adapter '{adapter.name}'.")
    print(f"Overall success: {overall['passed']}/{overall['total']} ({overall['rate']:.1%})")
    print(f"JSON: {args.output}")
    print(f"Markdown: {markdown_path}")
    return 0


def _report_command(args: argparse.Namespace) -> int:
    destination = report_from_json(args.run_json, args.output)
    print(f"Markdown: {destination}")
    return 0


def _build_adapter(args: argparse.Namespace) -> ModelAdapter:
    if args.adapter == "mock":
        return MockAgent()
    # The key is intentionally environment-only to keep it out of shell history.
    return OpenAICompatibleAdapter(
        model=args.model or os.getenv("TOOLCALL_BENCH_MODEL"),
        base_url=args.base_url or os.getenv("TOOLCALL_BENCH_BASE_URL"),
        timeout=args.timeout,
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

