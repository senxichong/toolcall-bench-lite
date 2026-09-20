"""Deterministic failure analysis and structured badcase artifacts."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping

from .schemas import RunReport, TaskRunResult


class FailureCategory(str, Enum):
    """Actionable, non-causal categories for agent evaluation failures."""

    WRONG_TOOL = "wrong_tool"
    WRONG_ARGUMENTS = "wrong_arguments"
    TOOL_FAILURE = "tool_failure"
    REASONING_FAILURE = "reasoning_failure"


@dataclass(frozen=True, slots=True)
class BadcaseRecord:
    """One failed task with category, evidence, and trace for review."""

    task_id: str
    primary_category: FailureCategory
    categories: tuple[FailureCategory, ...]
    reason: str
    expected_tool: str | None
    actual_tool: str | None
    expected_arguments: dict[str, Any]
    actual_arguments: dict[str, Any] | None
    expected_answer: Any
    actual_answer: str | None
    errors: tuple[str, ...]
    trace: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "primary_category": self.primary_category.value,
            "categories": [category.value for category in self.categories],
            "reason": self.reason,
            "expected_tool": self.expected_tool,
            "actual_tool": self.actual_tool,
            "expected_arguments": self.expected_arguments,
            "actual_arguments": self.actual_arguments,
            "expected_answer": self.expected_answer,
            "actual_answer": self.actual_answer,
            "errors": list(self.errors),
            "trace": list(self.trace),
        }


@dataclass(frozen=True, slots=True)
class BadcaseReport:
    """Structured, serializable failure summary for one model/run."""

    model: str
    benchmark: str | None
    total_tasks: int
    failed_tasks: int
    category_counts: dict[str, int]
    badcases: tuple[BadcaseRecord, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "model": self.model,
            "benchmark": self.benchmark,
            "total_tasks": self.total_tasks,
            "failed_tasks": self.failed_tasks,
            "category_counts": self.category_counts,
            "badcases": [record.to_dict() for record in self.badcases],
        }


class RuleBasedBadcaseAnalyzer:
    """Classify observed failures without making unsupported causal claims.

    Multiple categories may apply. The first is the primary category used for
    counts; priorities favor upstream decision errors over downstream effects.
    """

    def analyze_result(self, result: TaskRunResult) -> BadcaseRecord | None:
        evaluation = result.evaluation
        if evaluation.overall_success:
            return None

        categories: list[FailureCategory] = []
        reasons: list[str] = []
        if not evaluation.tool_selection_correct:
            categories.append(FailureCategory.WRONG_TOOL)
            reasons.append("Selected tool differs from the benchmark's expected tool.")
        if not evaluation.arguments_correct and evaluation.tool_selection_correct:
            categories.append(FailureCategory.WRONG_ARGUMENTS)
            reasons.append("Selected tool arguments differ from the expected arguments.")
        if not evaluation.schema_valid and evaluation.tool_selection_correct:
            if result.turn.tool_call is not None:
                if FailureCategory.WRONG_ARGUMENTS not in categories:
                    categories.append(FailureCategory.WRONG_ARGUMENTS)
                reasons.append("Tool call arguments or response structure violate the schema.")
            else:
                categories.append(FailureCategory.REASONING_FAILURE)
                reasons.append("Model response is malformed or missing a final answer.")
        if not evaluation.execution_success and result.execution is not None:
            categories.append(FailureCategory.TOOL_FAILURE)
            reasons.append("Tool execution did not complete successfully.")
        if not evaluation.final_answer_correct:
            categories.append(FailureCategory.REASONING_FAILURE)
            reasons.append("Final answer did not match the configured answer policy.")
        if not categories:
            categories.append(FailureCategory.REASONING_FAILURE)
            reasons.append("The run failed an evaluation check not covered by a narrower rule.")

        unique_categories = tuple(dict.fromkeys(categories))
        trace_steps: Iterable[Any] = getattr(result, "trace", ())
        trace = tuple(
            step.to_dict() if hasattr(step, "to_dict") else dict(step)
            for step in trace_steps
        )
        call = result.turn.tool_call
        return BadcaseRecord(
            task_id=result.task_id,
            primary_category=unique_categories[0],
            categories=unique_categories,
            reason=" ".join(dict.fromkeys(reasons)),
            expected_tool=result.task.expected_tool,
            actual_tool=None if call is None else call.name,
            expected_arguments=result.task.expected_arguments,
            actual_arguments=None if call is None else call.arguments,
            expected_answer=result.task.expected_answer,
            actual_answer=result.final_answer,
            errors=tuple(evaluation.errors),
            trace=trace,
        )

    def analyze(self, report: RunReport) -> BadcaseReport:
        records = tuple(
            record
            for result in report.results
            if (record := self.analyze_result(result)) is not None
        )
        counts = Counter(record.primary_category.value for record in records)
        return BadcaseReport(
            model=report.model,
            benchmark=report.benchmark,
            total_tasks=report.total_tasks,
            failed_tasks=len(records),
            category_counts=dict(sorted(counts.items())),
            badcases=records,
        )


def analyze_badcases(report: RunReport) -> BadcaseReport:
    """Convenience entry point for deterministic badcase classification."""

    return RuleBasedBadcaseAnalyzer().analyze(report)


def write_badcase_report(report: BadcaseReport, path: str | Path) -> Path:
    """Persist a structured badcase report as UTF-8 JSON."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report.to_dict(), ensure_ascii=False, indent=2, allow_nan=False)
    destination.write_text(payload + "\n", encoding="utf-8")
    return destination


def load_badcase_report(path: str | Path) -> Mapping[str, Any]:
    """Read a report for dashboard/API consumers without losing schema fields."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load badcase report {source}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("unsupported badcase report format")
    return payload


__all__ = [
    "BadcaseRecord",
    "BadcaseReport",
    "FailureCategory",
    "RuleBasedBadcaseAnalyzer",
    "analyze_badcases",
    "load_badcase_report",
    "write_badcase_report",
]

