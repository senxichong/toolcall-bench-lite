"""Independent metrics for tool choice, arguments, execution, and final answers."""

from __future__ import annotations

import ast
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from .schemas import AgentTurn, BenchmarkTask, TaskEvaluation, ToolExecutionResult
from .tools import DEFAULT_REGISTRY, ToolRegistry, ToolRegistryError


class ToolSelectionAccuracy:
    """Whether the model selected the gold tool, including correct no-call decisions."""

    @staticmethod
    def evaluate(task: BenchmarkTask, turn: AgentTurn) -> bool:
        actual = None if turn.tool_call is None else turn.tool_call.name
        return actual == task.expected_tool


class ArgumentMatch:
    """Structural argument comparison with numeric and expression normalization."""

    @staticmethod
    def evaluate(task: BenchmarkTask, turn: AgentTurn) -> bool:
        if task.expected_tool is None:
            return turn.tool_call is None
        if turn.tool_call is None or turn.tool_call.name != task.expected_tool:
            return False
        return arguments_match(task.expected_arguments, turn.tool_call.arguments)


class SchemaValidity:
    """Whether the response and any requested call satisfy the advertised schema."""

    @staticmethod
    def evaluate(task: BenchmarkTask, turn: AgentTurn, registry: ToolRegistry) -> bool:
        if not turn.schema_valid:
            return False
        if turn.tool_call is None:
            return turn.content is not None
        if turn.tool_call.name not in task.available_tools:
            return False
        try:
            return not registry.validate_arguments(turn.tool_call.name, turn.tool_call.arguments)
        except ToolRegistryError:
            return False


class ExecutionSuccess:
    """Whether a requested tool ran successfully, or no call was correctly expected."""

    @staticmethod
    def evaluate(
        task: BenchmarkTask,
        turn: AgentTurn,
        execution: ToolExecutionResult | None,
    ) -> bool:
        if turn.tool_call is None:
            return task.expected_tool is None
        return execution is not None and execution.success


class FinalAnswerAccuracy:
    """Whether the final answer matches the task's configured comparison policy."""

    @staticmethod
    def evaluate(task: BenchmarkTask, answer: str | None) -> bool:
        return answer_matches(task.expected_answer, answer, task.answer_match)


class Evaluator:
    """Compute all built-in metrics for one completed task trace."""

    def __init__(self, registry: ToolRegistry = DEFAULT_REGISTRY) -> None:
        self.registry = registry

    def evaluate(
        self,
        task: BenchmarkTask,
        turn: AgentTurn,
        execution: ToolExecutionResult | None = None,
        final_turn: AgentTurn | None = None,
    ) -> TaskEvaluation:
        answer = final_turn.content if final_turn is not None else turn.content
        selection = ToolSelectionAccuracy.evaluate(task, turn)
        arguments = ArgumentMatch.evaluate(task, turn)
        schema = SchemaValidity.evaluate(task, turn, self.registry)
        if final_turn is not None:
            schema = (
                schema
                and final_turn.schema_valid
                and final_turn.tool_call is None
                and final_turn.content is not None
            )
        execution_ok = ExecutionSuccess.evaluate(task, turn, execution)
        final_answer = (
            final_turn is None or final_turn.tool_call is None
        ) and FinalAnswerAccuracy.evaluate(task, answer)

        actual_tool = None if turn.tool_call is None else turn.tool_call.name
        actual_arguments = None if turn.tool_call is None else turn.tool_call.arguments
        details: dict[str, Any] = {
            "expected_tool": task.expected_tool,
            "actual_tool": actual_tool,
            "expected_arguments": task.expected_arguments,
            "actual_arguments": actual_arguments,
            "expected_answer": task.expected_answer,
            "actual_answer": answer,
            "answer_match": task.answer_match,
        }
        errors: list[str] = []
        if turn.error:
            errors.append(turn.error)
        if turn.tool_call is not None:
            try:
                schema_errors = self.registry.validate_arguments(
                    turn.tool_call.name, turn.tool_call.arguments
                )
            except ToolRegistryError as exc:
                schema_errors = [str(exc)]
            if schema_errors:
                details["schema_errors"] = schema_errors
        if execution is not None and execution.error:
            errors.append(execution.error)
        if final_turn is not None and final_turn.error:
            errors.append(f"final response: {final_turn.error}")

        return TaskEvaluation(
            task_id=task.task_id,
            tool_selection_correct=selection,
            arguments_correct=arguments,
            schema_valid=schema,
            execution_success=execution_ok,
            final_answer_correct=final_answer,
            details=details,
            errors=errors,
        )


def arguments_match(expected: Any, actual: Any, field: str = "") -> bool:
    """Recursively compare JSON-like arguments without unsafe type coercion."""

    if isinstance(expected, bool) or isinstance(actual, bool):
        return type(expected) is type(actual) and expected == actual
    if _is_number(expected) and _is_number(actual):
        return math.isclose(float(expected), float(actual), rel_tol=1e-9, abs_tol=1e-9)
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        if set(expected) != set(actual):
            return False
        return all(arguments_match(expected[key], actual[key], key) for key in expected)
    if _is_sequence(expected) and _is_sequence(actual):
        return len(expected) == len(actual) and all(
            arguments_match(left, right, field) for left, right in zip(expected, actual)
        )
    if isinstance(expected, str) and isinstance(actual, str):
        if field == "expression":
            return _expressions_match(expected, actual)
        return _normalize_text(expected) == _normalize_text(actual)
    return type(expected) is type(actual) and expected == actual


def answer_matches(expected: Any, actual: str | None, mode: str = "auto") -> bool:
    """Compare final answers using exact, JSON, numeric, contains, or auto mode."""

    if actual is None:
        return False
    selected_mode = mode
    if selected_mode == "auto":
        if isinstance(expected, (dict, list)):
            selected_mode = "json"
        elif _is_number(expected):
            selected_mode = "numeric"
        else:
            selected_mode = "normalized"

    if selected_mode == "json":
        try:
            decoded = json.loads(actual) if isinstance(actual, str) else actual
        except (json.JSONDecodeError, TypeError):
            return False
        return arguments_match(expected, decoded)
    if selected_mode == "numeric":
        try:
            return math.isclose(float(expected), float(actual.strip()), rel_tol=1e-9, abs_tol=1e-9)
        except (TypeError, ValueError):
            return False
    expected_text = str(expected)
    if selected_mode == "exact":
        return expected_text.strip() == actual.strip()
    if selected_mode == "contains":
        return _normalize_text(expected_text) in _normalize_text(actual)
    return _normalize_text(expected_text) == _normalize_text(actual)


def _expressions_match(expected: str, actual: str) -> bool:
    try:
        expected_tree = ast.dump(ast.parse(expected, mode="eval"), include_attributes=False)
        actual_tree = ast.dump(ast.parse(actual, mode="eval"), include_attributes=False)
    except SyntaxError:
        return re.sub(r"\s+", "", expected) == re.sub(r"\s+", "", actual)
    return expected_tree == actual_tree


def _normalize_text(value: str) -> str:
    return " ".join(value.strip().split()).casefold()


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def evaluate(
    task: BenchmarkTask,
    turn: AgentTurn,
    execution: ToolExecutionResult | None = None,
    final_turn: AgentTurn | None = None,
    registry: ToolRegistry = DEFAULT_REGISTRY,
) -> TaskEvaluation:
    """Functional shortcut for ``Evaluator(registry).evaluate(...)``."""

    return Evaluator(registry).evaluate(task, turn, execution, final_turn)
