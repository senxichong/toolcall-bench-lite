"""Typed data models shared by the benchmark runner and reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping


class SchemaValidationError(ValueError):
    """Raised when benchmark data does not match the expected shape."""


def _require_string(value: Any, field_name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise SchemaValidationError(f"{field_name} must be a string")
    if not allow_empty and not value.strip():
        raise SchemaValidationError(f"{field_name} must not be empty")
    return value


def _require_mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SchemaValidationError(f"{field_name} must be an object")
    result = dict(value)
    if not all(isinstance(key, str) for key in result):
        raise SchemaValidationError(f"{field_name} keys must be strings")
    return result


def _reject_unknown(data: Mapping[str, Any], allowed: set[str], model: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        names = ", ".join(unknown)
        raise SchemaValidationError(f"unknown {model} field(s): {names}")


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """Description and JSON Schema for one callable tool."""

    name: str
    description: str
    parameters: dict[str, Any]

    def __post_init__(self) -> None:
        _require_string(self.name, "tool name")
        _require_string(self.description, "tool description")
        schema = _require_mapping(self.parameters, "tool parameters")
        if schema.get("type") != "object":
            raise SchemaValidationError("tool parameters must be an object schema")
        if not isinstance(schema.get("properties", {}), Mapping):
            raise SchemaValidationError("tool parameter properties must be an object")
        required = schema.get("required", [])
        if not isinstance(required, list) or not all(
            isinstance(item, str) for item in required
        ):
            raise SchemaValidationError("tool parameter required must be a string list")

    @property
    def json_schema(self) -> dict[str, Any]:
        """Return the parameter schema (a shallow copy)."""

        return dict(self.parameters)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    def to_openai_tool(self) -> dict[str, Any]:
        """Return the shape accepted by OpenAI-compatible chat APIs."""

        return {"type": "function", "function": self.to_dict()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ToolSpec:
        raw = _require_mapping(data, "tool spec")
        if raw.get("type") == "function" and "function" in raw:
            raw = _require_mapping(raw["function"], "tool function")
        _reject_unknown(raw, {"name", "description", "parameters"}, "tool spec")
        return cls(
            name=_require_string(raw.get("name"), "tool name"),
            description=_require_string(raw.get("description"), "tool description"),
            parameters=_require_mapping(raw.get("parameters"), "tool parameters"),
        )


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A model-requested tool invocation."""

    name: str
    arguments: dict[str, Any]
    id: str | None = None

    def __post_init__(self) -> None:
        _require_string(self.name, "tool call name")
        _require_mapping(self.arguments, "tool call arguments")
        if self.id is not None:
            _require_string(self.id, "tool call id")

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"name": self.name, "arguments": self.arguments}
        if self.id is not None:
            data["id"] = self.id
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ToolCall:
        raw = _require_mapping(data, "tool call")
        _reject_unknown(raw, {"name", "arguments", "id"}, "tool call")
        call_id = raw.get("id")
        return cls(
            name=_require_string(raw.get("name"), "tool call name"),
            arguments=_require_mapping(raw.get("arguments"), "tool call arguments"),
            id=None if call_id is None else _require_string(call_id, "tool call id"),
        )


@dataclass(frozen=True, slots=True)
class AgentTurn:
    """Normalized output from a model adapter."""

    content: str | None = None
    tool_call: ToolCall | None = None
    raw: Any = None
    schema_valid: bool = True
    error: str | None = None

    def __post_init__(self) -> None:
        if self.content is not None and not isinstance(self.content, str):
            raise SchemaValidationError("agent turn content must be a string or null")
        if not isinstance(self.schema_valid, bool):
            raise SchemaValidationError("agent turn schema_valid must be a boolean")
        if self.error is not None and not isinstance(self.error, str):
            raise SchemaValidationError("agent turn error must be a string or null")

    @property
    def final_answer(self) -> str | None:
        """Alias used by evaluators after tool execution."""

        return self.content

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "tool_call": None if self.tool_call is None else self.tool_call.to_dict(),
            "raw": self.raw,
            "schema_valid": self.schema_valid,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AgentTurn:
        raw = _require_mapping(data, "agent turn")
        _reject_unknown(
            raw,
            {"content", "tool_call", "raw", "schema_valid", "error"},
            "agent turn",
        )
        call = raw.get("tool_call")
        return cls(
            content=raw.get("content"),
            tool_call=None if call is None else ToolCall.from_dict(call),
            raw=raw.get("raw"),
            schema_valid=raw.get("schema_valid", True),
            error=raw.get("error"),
        )


@dataclass(frozen=True, slots=True)
class ToolExecutionResult:
    """Result of a local tool invocation; failures are data, not exceptions."""

    success: bool
    value: Any = None
    error: str | None = None
    tool_name: str | None = None
    arguments: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.success, bool):
            raise SchemaValidationError("execution success must be a boolean")
        if self.error is not None and not isinstance(self.error, str):
            raise SchemaValidationError("execution error must be a string or null")
        if self.arguments is not None:
            _require_mapping(self.arguments, "execution arguments")

    @property
    def output(self) -> Any:
        """Compatibility alias for adapters that call tool values outputs."""

        return self.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "value": self.value,
            "error": self.error,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ToolExecutionResult:
        raw = _require_mapping(data, "execution result")
        _reject_unknown(
            raw,
            {"success", "value", "error", "tool_name", "arguments"},
            "execution result",
        )
        if "success" not in raw:
            raise SchemaValidationError("execution result is missing success")
        return cls(
            success=raw["success"],
            value=raw.get("value"),
            error=raw.get("error"),
            tool_name=raw.get("tool_name"),
            arguments=raw.get("arguments"),
        )


@dataclass(frozen=True, slots=True)
class BenchmarkTask:
    """One deterministic benchmark example."""

    task_id: str
    prompt: str
    available_tools: list[str]
    expected_tool: str | None
    expected_arguments: dict[str, Any]
    expected_answer: Any
    answer_match: str = "auto"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_string(self.task_id, "task_id")
        _require_string(self.prompt, "prompt")
        if not isinstance(self.available_tools, list) or not all(
            isinstance(name, str) and bool(name.strip()) for name in self.available_tools
        ):
            raise SchemaValidationError("available_tools must be a list of tool names")
        if len(set(self.available_tools)) != len(self.available_tools):
            raise SchemaValidationError("available_tools must not contain duplicates")
        if self.expected_tool is not None:
            _require_string(self.expected_tool, "expected_tool")
            if self.expected_tool not in self.available_tools:
                raise SchemaValidationError(
                    "expected_tool must be present in available_tools"
                )
        _require_mapping(self.expected_arguments, "expected_arguments")
        if self.expected_tool is None and self.expected_arguments:
            raise SchemaValidationError(
                "expected_arguments must be empty when expected_tool is null"
            )
        if self.answer_match not in {"auto", "exact", "json", "numeric", "contains"}:
            raise SchemaValidationError(
                "answer_match must be auto, exact, json, numeric, or contains"
            )
        _require_mapping(self.metadata, "metadata")

    @property
    def requires_tool(self) -> bool:
        return self.expected_tool is not None

    def to_dict(self) -> dict[str, Any]:
        data = {
            "task_id": self.task_id,
            "prompt": self.prompt,
            "available_tools": self.available_tools,
            "expected_tool": self.expected_tool,
            "expected_arguments": self.expected_arguments,
            "expected_answer": self.expected_answer,
            "answer_match": self.answer_match,
        }
        if self.metadata:
            data["metadata"] = self.metadata
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BenchmarkTask:
        raw = _require_mapping(data, "benchmark task")
        required = {
            "task_id",
            "prompt",
            "available_tools",
            "expected_tool",
            "expected_arguments",
            "expected_answer",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise SchemaValidationError(
                f"benchmark task is missing field(s): {', '.join(missing)}"
            )
        _reject_unknown(raw, required | {"answer_match", "metadata"}, "benchmark task")
        tools = raw["available_tools"]
        if not isinstance(tools, list):
            raise SchemaValidationError("available_tools must be a list")
        expected_tool = raw["expected_tool"]
        if expected_tool is not None and not isinstance(expected_tool, str):
            raise SchemaValidationError("expected_tool must be a string or null")
        return cls(
            task_id=_require_string(raw["task_id"], "task_id"),
            prompt=_require_string(raw["prompt"], "prompt"),
            available_tools=list(tools),
            expected_tool=expected_tool,
            expected_arguments=_require_mapping(
                raw["expected_arguments"], "expected_arguments"
            ),
            expected_answer=raw["expected_answer"],
            answer_match=raw.get("answer_match", "auto"),
            metadata=_require_mapping(raw.get("metadata", {}), "metadata"),
        )


@dataclass(frozen=True, slots=True)
class TaskEvaluation:
    """Boolean metric outcomes for a single task."""

    task_id: str
    tool_selection_correct: bool
    arguments_correct: bool
    schema_valid: bool
    execution_success: bool
    final_answer_correct: bool
    overall_success: bool | None = None
    details: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        _require_string(self.task_id, "evaluation task_id")
        metric_values = (
            self.tool_selection_correct,
            self.arguments_correct,
            self.schema_valid,
            self.execution_success,
            self.final_answer_correct,
        )
        if not all(isinstance(value, bool) for value in metric_values):
            raise SchemaValidationError("evaluation metric values must be booleans")
        if self.overall_success is None:
            object.__setattr__(self, "overall_success", all(metric_values))
        elif not isinstance(self.overall_success, bool):
            raise SchemaValidationError("overall_success must be a boolean")
        _require_mapping(self.details, "evaluation details")
        if not isinstance(self.errors, list) or not all(
            isinstance(item, str) for item in self.errors
        ):
            raise SchemaValidationError("evaluation errors must be a string list")

    @property
    def argument_match(self) -> bool:
        """Compatibility alias matching the public metric name."""

        return self.arguments_correct

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "tool_selection_correct": self.tool_selection_correct,
            "arguments_correct": self.arguments_correct,
            "schema_valid": self.schema_valid,
            "execution_success": self.execution_success,
            "final_answer_correct": self.final_answer_correct,
            "overall_success": self.overall_success,
            "details": self.details,
            "errors": self.errors,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TaskEvaluation:
        raw = _require_mapping(data, "task evaluation")
        if "arguments_correct" not in raw and "argument_match" in raw:
            raw["arguments_correct"] = raw.pop("argument_match")
        required = {
            "task_id",
            "tool_selection_correct",
            "arguments_correct",
            "schema_valid",
            "execution_success",
            "final_answer_correct",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise SchemaValidationError(
                f"task evaluation is missing field(s): {', '.join(missing)}"
            )
        _reject_unknown(
            raw,
            required | {"overall_success", "details", "errors"},
            "task evaluation",
        )
        return cls(
            task_id=raw["task_id"],
            tool_selection_correct=raw["tool_selection_correct"],
            arguments_correct=raw["arguments_correct"],
            schema_valid=raw["schema_valid"],
            execution_success=raw["execution_success"],
            final_answer_correct=raw["final_answer_correct"],
            overall_success=raw.get("overall_success"),
            details=_require_mapping(raw.get("details", {}), "evaluation details"),
            errors=list(raw.get("errors", [])),
        )


@dataclass(frozen=True, slots=True)
class TaskRunResult:
    """Complete trace and evaluation for one task."""

    task: BenchmarkTask
    turn: AgentTurn
    execution: ToolExecutionResult | None
    evaluation: TaskEvaluation
    final_turn: AgentTurn | None = None
    latency_ms: float = 0.0

    def __post_init__(self) -> None:
        if self.task.task_id != self.evaluation.task_id:
            raise SchemaValidationError("task and evaluation task_id must match")
        if isinstance(self.latency_ms, bool) or not isinstance(
            self.latency_ms, (int, float)
        ):
            raise SchemaValidationError("latency_ms must be a number")
        if self.latency_ms < 0:
            raise SchemaValidationError("latency_ms must not be negative")

    @property
    def task_id(self) -> str:
        return self.task.task_id

    @property
    def final_answer(self) -> str | None:
        """Return the post-tool content when present, otherwise first-turn content."""

        if self.final_turn is not None:
            return self.final_turn.content
        return self.turn.content

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task.to_dict(),
            "turn": self.turn.to_dict(),
            "execution": None if self.execution is None else self.execution.to_dict(),
            "evaluation": self.evaluation.to_dict(),
            "final_turn": None if self.final_turn is None else self.final_turn.to_dict(),
            "latency_ms": self.latency_ms,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TaskRunResult:
        raw = _require_mapping(data, "task run result")
        _reject_unknown(
            raw,
            {
                "task",
                "turn",
                "execution",
                "evaluation",
                "final_turn",
                "latency_ms",
            },
            "task run result",
        )
        for required in ("task", "turn", "evaluation"):
            if required not in raw:
                raise SchemaValidationError(
                    f"task run result is missing field: {required}"
                )
        execution = raw.get("execution")
        final_turn = raw.get("final_turn")
        return cls(
            task=BenchmarkTask.from_dict(raw["task"]),
            turn=AgentTurn.from_dict(raw["turn"]),
            execution=(
                None if execution is None else ToolExecutionResult.from_dict(execution)
            ),
            evaluation=TaskEvaluation.from_dict(raw["evaluation"]),
            final_turn=None if final_turn is None else AgentTurn.from_dict(final_turn),
            latency_ms=raw.get("latency_ms", 0.0),
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, slots=True)
class RunReport:
    """Serializable collection of task results plus computed aggregate metrics."""

    model: str
    results: list[TaskRunResult]
    created_at: str = field(default_factory=_utc_now)
    benchmark: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_string(self.model, "model")
        if not isinstance(self.results, list) or not all(
            isinstance(result, TaskRunResult) for result in self.results
        ):
            raise SchemaValidationError("results must be a list of TaskRunResult")
        _require_string(self.created_at, "created_at")
        if self.benchmark is not None:
            _require_string(self.benchmark, "benchmark")
        _require_mapping(self.metadata, "run metadata")

    @property
    def total_tasks(self) -> int:
        return len(self.results)

    def summary(self) -> dict[str, Any]:
        """Compute counts and rates; no values are stored or fabricated."""

        total = self.total_tasks
        metric_fields = {
            "tool_selection_accuracy": "tool_selection_correct",
            "argument_match": "arguments_correct",
            "schema_validity": "schema_valid",
            "execution_success": "execution_success",
            "final_answer_accuracy": "final_answer_correct",
            "overall_success": "overall_success",
        }
        summary: dict[str, Any] = {"total_tasks": total}
        for output_name, field_name in metric_fields.items():
            passed = sum(
                bool(getattr(result.evaluation, field_name)) for result in self.results
            )
            summary[output_name] = {
                "passed": passed,
                "total": total,
                "rate": passed / total if total else 0.0,
            }
        return summary

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "created_at": self.created_at,
            "benchmark": self.benchmark,
            "metadata": self.metadata,
            "summary": self.summary(),
            "results": [result.to_dict() for result in self.results],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RunReport:
        raw = _require_mapping(data, "run report")
        _reject_unknown(
            raw,
            {"model", "created_at", "benchmark", "metadata", "summary", "results"},
            "run report",
        )
        if "model" not in raw or "results" not in raw:
            raise SchemaValidationError("run report requires model and results")
        results = raw["results"]
        if not isinstance(results, list):
            raise SchemaValidationError("run report results must be a list")
        return cls(
            model=raw["model"],
            results=[TaskRunResult.from_dict(item) for item in results],
            created_at=raw.get("created_at", _utc_now()),
            benchmark=raw.get("benchmark"),
            metadata=_require_mapping(raw.get("metadata", {}), "run metadata"),
        )
