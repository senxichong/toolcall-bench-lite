"""Human-readable Markdown rendering for benchmark run JSON."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .runner import load_run_report
from .schemas import RunReport

_METRIC_LABELS = {
    "tool_selection_accuracy": "Tool Selection Accuracy",
    "argument_match": "Argument Match",
    "schema_validity": "Schema Validity",
    "execution_success": "Execution Success",
    "final_answer_accuracy": "Final Answer Accuracy",
    "overall_success": "Overall Success",
}


def render_markdown(report: RunReport | Mapping[str, Any]) -> str:
    """Render a report from typed data or its JSON-compatible dictionary."""

    typed = report if isinstance(report, RunReport) else RunReport.from_dict(report)
    summary = typed.summary()
    lines = [
        "# ToolCall-Bench-Lite Report",
        "",
        f"- Model: `{_escape_inline(typed.model)}`",
        f"- Created: `{_escape_inline(typed.created_at)}`",
        f"- Benchmark: `{_escape_inline(typed.benchmark or 'in-memory')}`",
        f"- Tasks: **{typed.total_tasks}**",
        "",
        "## Summary",
        "",
        "| Metric | Passed | Total | Rate |",
        "|---|---:|---:|---:|",
    ]
    for key, label in _METRIC_LABELS.items():
        value = summary[key]
        lines.append(
            f"| {label} | {value['passed']} | {value['total']} | {value['rate']:.1%} |"
        )

    lines.extend(
        [
            "",
            "## Per-task Results",
            "",
            "| Task | Tool | Arguments | Schema | Execution | Final | Overall |",
            "|---|:---:|:---:|:---:|:---:|:---:|:---:|",
        ]
    )
    for result in typed.results:
        evaluation = result.evaluation
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape_table(result.task_id),
                    _mark(evaluation.tool_selection_correct),
                    _mark(evaluation.arguments_correct),
                    _mark(evaluation.schema_valid),
                    _mark(evaluation.execution_success),
                    _mark(evaluation.final_answer_correct),
                    _mark(bool(evaluation.overall_success)),
                ]
            )
            + " |"
        )

    failures = [result for result in typed.results if not result.evaluation.overall_success]
    if failures:
        lines.extend(["", "## Failure Details", ""])
        for result in failures:
            evaluation = result.evaluation
            details = evaluation.details
            lines.extend(
                [
                    f"### `{_escape_inline(result.task_id)}`",
                    "",
                    f"- Expected tool: `{_json_inline(details.get('expected_tool'))}`",
                    f"- Actual tool: `{_json_inline(details.get('actual_tool'))}`",
                    f"- Expected arguments: `{_json_inline(details.get('expected_arguments'))}`",
                    f"- Actual arguments: `{_json_inline(details.get('actual_arguments'))}`",
                    f"- Expected answer: `{_json_inline(details.get('expected_answer'))}`",
                    f"- Actual answer: `{_json_inline(details.get('actual_answer'))}`",
                ]
            )
            if evaluation.errors:
                lines.append(f"- Errors: {_escape_inline('; '.join(evaluation.errors))}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_markdown_report(
    report: RunReport | Mapping[str, Any], path: str | Path
) -> Path:
    """Write a rendered Markdown report and return its path."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_markdown(report), encoding="utf-8")
    return destination


def report_from_json(source: str | Path, output: str | Path | None = None) -> Path:
    """Load run JSON and write Markdown beside it unless an output is supplied."""

    source_path = Path(source)
    destination = Path(output) if output is not None else source_path.with_suffix(".md")
    return write_markdown_report(load_run_report(source_path), destination)


def _mark(value: bool) -> str:
    return "✅" if value else "❌"


def _escape_table(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _escape_inline(value: Any) -> str:
    return str(value).replace("`", "\\`").replace("\n", " ")


def _json_inline(value: Any) -> str:
    return _escape_inline(json.dumps(value, ensure_ascii=False, separators=(",", ":")))

