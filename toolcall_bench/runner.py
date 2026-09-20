"""Single-tool benchmark orchestration."""

from __future__ import annotations

import json
import time
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

from .agent import (
    LLMBackend,
    LocalModelBackend,
    MockAgent,
    ModelAdapterError,
    OpenAICompatibleAdapter,
)
from .evaluator import Evaluator
from .parser import parse_agent_response
from .schemas import (
    AgentTurn,
    BenchmarkTask,
    RunReport,
    TaskRunResult,
    ToolExecutionResult,
    TraceStep,
)
from .tasks import load_tasks
from .tools import DEFAULT_REGISTRY, ToolRegistry


class BenchmarkRunner:
    """Run a benchmark with at most one tool invocation per task."""

    def __init__(
        self,
        adapter: LLMBackend | None = None,
        registry: ToolRegistry = DEFAULT_REGISTRY,
        evaluator: Evaluator | None = None,
    ) -> None:
        self.adapter = adapter or MockAgent()
        self.registry = registry
        self.evaluator = evaluator or Evaluator(registry)

    def run(
        self,
        benchmark: str | Path | Iterable[BenchmarkTask],
        output_path: str | Path | None = None,
    ) -> RunReport:
        if isinstance(benchmark, (str, Path)):
            tasks = load_tasks(benchmark)
            benchmark_name = str(benchmark)
        else:
            tasks = list(benchmark)
            benchmark_name = None
        if not tasks:
            raise ValueError("benchmark must contain at least one task")

        results = [self.run_task(task) for task in tasks]
        model_name = getattr(self.adapter, "model", None) or self.adapter.name
        report = RunReport(
            model=str(model_name),
            results=results,
            benchmark=benchmark_name,
            metadata={"adapter": self.adapter.name, "single_tool_step": True},
        )
        if output_path is not None:
            write_run_report(report, output_path)
        return report

    def run_task(self, task: BenchmarkTask) -> TaskRunResult:
        started = time.perf_counter()
        specs = self.registry.specs(task.available_tools)
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "Use at most one available tool when it is needed. "
                    "Never invent missing required arguments. After a tool result, "
                    "answer the user's request directly."
                ),
            },
            {"role": "user", "content": task.prompt},
        ]
        trace: list[TraceStep] = []
        first_input = deepcopy(messages)
        turn, model_latency_ms = self._generate_turn_timed(messages, specs)
        trace.append(
            TraceStep(
                step_index=0,
                kind="model",
                user_input=task.prompt,
                input_messages=first_input,
                model_output=turn.raw,
                selected_tool=None if turn.tool_call is None else turn.tool_call.name,
                tool_arguments=None if turn.tool_call is None else deepcopy(turn.tool_call.arguments),
                latency_ms=model_latency_ms,
                token_usage=turn.token_usage,
                error=turn.error,
            )
        )
        execution: ToolExecutionResult | None = None
        final_turn: AgentTurn | None = None

        if turn.tool_call is not None:
            call = turn.tool_call
            tool_started = time.perf_counter()
            if call.name not in task.available_tools:
                execution = ToolExecutionResult(
                    success=False,
                    error=f"tool was not advertised for this task: {call.name}",
                    tool_name=call.name,
                    arguments=call.arguments,
                )
            else:
                execution = self.registry.execute(call.name, call.arguments)
            execution = _sanitize_execution_error(execution)
            trace.append(
                TraceStep(
                    step_index=len(trace),
                    kind="tool",
                    user_input=task.prompt,
                    selected_tool=call.name,
                    tool_arguments=deepcopy(call.arguments),
                    tool_response=execution,
                    latency_ms=round((time.perf_counter() - tool_started) * 1000, 3),
                    error=execution.error,
                )
            )
            messages.extend(
                [
                    _assistant_tool_message(turn),
                    {
                        "role": "tool",
                        "tool_call_id": call.id or "tool_call_1",
                        "name": call.name,
                        "content": json.dumps(
                            {
                                "name": call.name,
                                "success": execution.success,
                                "value": execution.value,
                                "error": execution.error,
                            },
                            ensure_ascii=False,
                        ),
                    },
                ]
            )
            final_input = deepcopy(messages)
            final_turn, final_latency_ms = self._generate_turn_timed(messages, specs)
            if final_turn.tool_call is not None:
                message = "multiple tool steps are outside the lite runner's scope"
                combined = f"{final_turn.error}; {message}" if final_turn.error else message
                final_turn = replace(final_turn, schema_valid=False, error=combined)
            trace.append(
                TraceStep(
                    step_index=len(trace),
                    kind="model",
                    user_input=task.prompt,
                    input_messages=final_input,
                    model_output=final_turn.raw,
                    selected_tool=(
                        None if final_turn.tool_call is None else final_turn.tool_call.name
                    ),
                    tool_arguments=(
                        None if final_turn.tool_call is None
                        else deepcopy(final_turn.tool_call.arguments)
                    ),
                    latency_ms=final_latency_ms,
                    token_usage=final_turn.token_usage,
                    error=final_turn.error,
                )
            )

        evaluation = self.evaluator.evaluate(task, turn, execution, final_turn)
        latency_ms = round((time.perf_counter() - started) * 1000, 3)
        return TaskRunResult(
            task=task,
            turn=turn,
            execution=execution,
            evaluation=evaluation,
            final_turn=final_turn,
            latency_ms=latency_ms,
            trace=trace,
        )

    def _generate_turn(self, messages: list[dict[str, Any]], specs: list[Any]) -> AgentTurn:
        """Compatibility helper for callers that only need the parsed turn."""

        return self._generate_turn_timed(messages, specs)[0]

    def _generate_turn_timed(
        self, messages: list[dict[str, Any]], specs: list[Any]
    ) -> tuple[AgentTurn, float]:
        started = time.perf_counter()
        try:
            raw = self.adapter.generate(messages, specs)
        except Exception as exc:  # One provider failure should not erase the whole run.
            safe_error = f"adapter error: {type(exc).__name__}"
            if type(self.adapter) in {OpenAICompatibleAdapter, LocalModelBackend, MockAgent} and isinstance(
                exc, ModelAdapterError
            ):
                safe_error += f": {exc}"
            turn = AgentTurn(
                raw=None,
                schema_valid=False,
                error=safe_error,
            )
        else:
            turn = parse_agent_response(raw)
        return turn, round((time.perf_counter() - started) * 1000, 3)


def _sanitize_execution_error(execution: ToolExecutionResult) -> ToolExecutionResult:
    """Avoid persisting arbitrary exception text emitted by third-party tools."""

    error = execution.error
    if error and ": " in error:
        exception_type, _detail = error.split(": ", 1)
        if exception_type.endswith(("Error", "Exception")) and exception_type.isidentifier():
            return replace(execution, error=exception_type)
    return execution


def _assistant_tool_message(turn: AgentTurn) -> dict[str, Any]:
    if turn.tool_call is None:
        raise ValueError("turn has no tool call")
    call = turn.tool_call
    return {
        "role": "assistant",
        "content": turn.content,
        "tool_calls": [
            {
                "id": call.id or "tool_call_1",
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments, ensure_ascii=False),
                },
            }
        ],
    }


def write_run_report(report: RunReport, path: str | Path) -> Path:
    """Persist a machine-readable report."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return destination


def load_run_report(path: str | Path) -> RunReport:
    """Load a report produced by :func:`write_run_report`."""

    source = Path(path)
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load run report {source}: {exc}") from exc
    return RunReport.from_dict(data)


def run_benchmark(
    benchmark: str | Path | Iterable[BenchmarkTask],
    adapter: LLMBackend | None = None,
    output_path: str | Path | None = None,
    registry: ToolRegistry = DEFAULT_REGISTRY,
) -> RunReport:
    """Functional shortcut for the common one-shot workflow."""

    return BenchmarkRunner(adapter=adapter, registry=registry).run(benchmark, output_path)
