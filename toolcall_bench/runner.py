"""Single-tool benchmark orchestration."""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

from .agent import MockAgent, ModelAdapter
from .evaluator import Evaluator
from .parser import parse_agent_response
from .schemas import (
    AgentTurn,
    BenchmarkTask,
    RunReport,
    TaskRunResult,
    ToolExecutionResult,
)
from .tasks import load_tasks
from .tools import DEFAULT_REGISTRY, ToolRegistry


class BenchmarkRunner:
    """Run a benchmark with at most one tool invocation per task."""

    def __init__(
        self,
        adapter: ModelAdapter | None = None,
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
        turn = self._generate_turn(messages, specs)
        execution: ToolExecutionResult | None = None
        final_turn: AgentTurn | None = None

        if turn.tool_call is not None:
            call = turn.tool_call
            if call.name not in task.available_tools:
                execution = ToolExecutionResult(
                    success=False,
                    error=f"tool was not advertised for this task: {call.name}",
                    tool_name=call.name,
                    arguments=call.arguments,
                )
            else:
                execution = self.registry.execute(call.name, call.arguments)
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
            final_turn = self._generate_turn(messages, specs)
            if final_turn.tool_call is not None:
                message = "multiple tool steps are outside the lite runner's scope"
                combined = f"{final_turn.error}; {message}" if final_turn.error else message
                final_turn = replace(final_turn, schema_valid=False, error=combined)

        evaluation = self.evaluator.evaluate(task, turn, execution, final_turn)
        latency_ms = round((time.perf_counter() - started) * 1000, 3)
        return TaskRunResult(
            task=task,
            turn=turn,
            execution=execution,
            evaluation=evaluation,
            final_turn=final_turn,
            latency_ms=latency_ms,
        )

    def _generate_turn(self, messages: list[dict[str, Any]], specs: list[Any]) -> AgentTurn:
        try:
            raw = self.adapter.generate(messages, specs)
        except Exception as exc:  # One provider failure should not erase the whole run.
            return AgentTurn(
                raw=None,
                schema_valid=False,
                error=f"adapter error: {type(exc).__name__}: {exc}",
            )
        return parse_agent_response(raw)


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
    adapter: ModelAdapter | None = None,
    output_path: str | Path | None = None,
    registry: ToolRegistry = DEFAULT_REGISTRY,
) -> RunReport:
    """Functional shortcut for the common one-shot workflow."""

    return BenchmarkRunner(adapter=adapter, registry=registry).run(benchmark, output_path)

