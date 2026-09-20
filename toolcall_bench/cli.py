"""Command-line interface for running benchmarks and rendering reports."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from . import __version__
from .agent import LocalModelBackend, MockAgent, ModelAdapter, OpenAICompatibleAdapter
from .badcases import analyze_badcases, write_badcase_report
from .datasets import DatasetManager
from .report import report_from_json, write_markdown_report
from .runner import BenchmarkRunner, write_run_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="toolcall-bench",
        description="Traceable evaluation for LLM tool/function calling.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    run_parser = commands.add_parser("run", help="run a JSONL benchmark")
    run_parser.add_argument("benchmark", nargs="?", type=Path, help="path to benchmark JSONL")
    run_parser.add_argument(
        "--adapter",
        choices=("mock", "openai-compatible", "local"),
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
    run_parser.add_argument(
        "--dataset-name",
        help="use a named versioned dataset instead of a benchmark path",
    )
    run_parser.add_argument("--dataset-version", help="version such as v000001 (latest by default)")
    run_parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=Path("datasets"),
        help="versioned dataset root (default: datasets)",
    )
    run_parser.add_argument(
        "--badcases",
        type=Path,
        help="structured badcase JSON path (default: JSON output with .badcases.json suffix)",
    )

    report_parser = commands.add_parser("report", help="render Markdown from run JSON")
    report_parser.add_argument("run_json", type=Path, help="path to run JSON")
    report_parser.add_argument("--output", type=Path, help="Markdown output path")

    datasets_parser = commands.add_parser("dataset", help="manage immutable benchmark versions")
    datasets_parser.add_argument(
        "--root", type=Path, default=Path("datasets"), help="dataset storage root"
    )
    dataset_commands = datasets_parser.add_subparsers(dest="dataset_command", required=True)
    import_parser = dataset_commands.add_parser("import", help="import JSONL or JSON-array tasks")
    import_parser.add_argument("name")
    import_parser.add_argument("source", type=Path)
    list_parser = dataset_commands.add_parser("list", help="list dataset names")
    list_parser.add_argument("--name", help="list versions for one dataset")
    export_parser = dataset_commands.add_parser("export", help="export a verified version")
    export_parser.add_argument("name")
    export_parser.add_argument("destination", type=Path)
    export_parser.add_argument("--version")
    export_parser.add_argument("--format", choices=("jsonl", "json"), default="jsonl")
    verify_parser = dataset_commands.add_parser("verify", help="verify version digest and count")
    verify_parser.add_argument("name")
    verify_parser.add_argument("--version")
    diff_parser = dataset_commands.add_parser("diff", help="compare stable task IDs")
    diff_parser.add_argument("name")
    diff_parser.add_argument("older")
    diff_parser.add_argument("newer")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run":
            return _run_command(args)
        if args.command == "report":
            return _report_command(args)
        return _dataset_command(args)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _run_command(args: argparse.Namespace) -> int:
    if bool(args.benchmark) == bool(args.dataset_name):
        raise ValueError("provide exactly one benchmark path or --dataset-name")
    adapter = _build_adapter(args)
    if args.dataset_name:
        dataset_manager = DatasetManager(args.datasets_dir)
        version = dataset_manager.get_version(args.dataset_name, args.dataset_version)
        benchmark = dataset_manager.load_version(args.dataset_name, version.version)
    else:
        version = None
        benchmark = args.benchmark
    report = BenchmarkRunner(adapter=adapter).run(benchmark)
    if version is not None:
        report = replace(
            report,
            benchmark=f"{version.name}/{version.version}",
            metadata={
                **report.metadata,
                "dataset_name": version.name,
                "dataset_version": version.version,
                "dataset_sha256": version.sha256,
            },
        )
    write_run_report(report, args.output)
    markdown_path = args.markdown or args.output.with_suffix(".md")
    write_markdown_report(report, markdown_path)
    badcases_path = args.badcases or args.output.with_suffix(".badcases.json")
    write_badcase_report(analyze_badcases(report), badcases_path)
    summary = report.summary()
    overall = summary["overall_success"]
    print(f"Completed {overall['total']} tasks with adapter '{adapter.name}'.")
    print(f"Overall success: {overall['passed']}/{overall['total']} ({overall['rate']:.1%})")
    print(f"JSON: {args.output}")
    print(f"Markdown: {markdown_path}")
    print(f"Badcases: {badcases_path}")
    if version is not None:
        print(f"Dataset: {version.name}/{version.version} (sha256:{version.sha256})")
    return 0


def _report_command(args: argparse.Namespace) -> int:
    destination = report_from_json(args.run_json, args.output)
    print(f"Markdown: {destination}")
    return 0


def _build_adapter(args: argparse.Namespace) -> ModelAdapter:
    if args.adapter == "mock":
        return MockAgent()
    if args.adapter == "local":
        model = args.model or os.getenv("TOOLCALL_BENCH_MODEL")
        if not model:
            raise ValueError("--model or TOOLCALL_BENCH_MODEL is required for local inference")
        return LocalModelBackend(model=model)
    # The key is intentionally environment-only to keep it out of shell history.
    return OpenAICompatibleAdapter(
        model=args.model or os.getenv("TOOLCALL_BENCH_MODEL"),
        base_url=args.base_url or os.getenv("TOOLCALL_BENCH_BASE_URL"),
        timeout=args.timeout,
    )


def _dataset_command(args: argparse.Namespace) -> int:
    manager = DatasetManager(args.root)
    action = args.dataset_command
    if action == "import":
        payload = manager.import_file(args.name, args.source).to_dict()
    elif action == "list":
        payload = (
            [item.to_dict() for item in manager.list_versions(args.name)]
            if args.name
            else manager.list_datasets()
        )
    elif action == "export":
        path = manager.export_version(
            args.name,
            args.destination,
            version=args.version,
            format=args.format,
        )
        payload = {"exported": str(path), "name": args.name, "version": args.version or "latest"}
    elif action == "verify":
        manifest = manager.get_version(args.name, args.version)
        manager.load_version(args.name, manifest.version)
        payload = {"verified": True, **manifest.to_dict()}
    else:
        payload = manager.diff_versions(args.name, args.older, args.newer)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
