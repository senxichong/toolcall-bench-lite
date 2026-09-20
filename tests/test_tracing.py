"""Regression tests for the persisted, provider-neutral execution trace."""

import json

from toolcall_bench.agent import MockAgent, ModelAdapter
from toolcall_bench.runner import BenchmarkRunner, load_run_report
from toolcall_bench.schemas import BenchmarkTask, RunReport, TaskRunResult


def _task():
    return BenchmarkTask(
        task_id="trace-calc",
        prompt="What is 2 + 2?",
        available_tools=["calculator"],
        expected_tool="calculator",
        expected_arguments={"expression": "2 + 2"},
        expected_answer="4",
        answer_match="exact",
    )


def test_trace_records_model_tool_and_final_model_events(tmp_path):
    report = BenchmarkRunner(MockAgent()).run([_task()], tmp_path / "run.json")
    result = report.results[0]
    assert [step.kind for step in result.trace] == ["model", "tool", "model"]
    assert [step.step_index for step in result.trace] == [0, 1, 2]
    assert all(step.user_input == _task().prompt for step in result.trace)
    assert result.trace[0].input_messages[-1] == {"role": "user", "content": _task().prompt}
    assert result.trace[0].selected_tool == "calculator"
    assert result.trace[0].tool_arguments == {"expression": "2 + 2"}
    assert result.trace[1].tool_response.success
    assert result.trace[1].tool_response.value == 4
    assert json.loads(result.trace[2].input_messages[-1]["content"])["value"] == 4
    assert result.trace[2].model_output == {"content": "4"}
    assert all(step.latency_ms >= 0 for step in result.trace)
    loaded = load_run_report(tmp_path / "run.json")
    assert loaded.results[0].trace == result.trace
    assert RunReport.from_dict(report.to_dict()).results[0].trace == result.trace


class UsageAdapter(ModelAdapter):
    name = "usage"

    def generate(self, messages, tools):
        if messages[-1]["role"] == "tool":
            return {
                "choices": [{"message": {"content": "4"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            }
        return {
            "choices": [{"message": {"tool_calls": [{
                "type": "function",
                "function": {"name": "calculator", "arguments": '{"expression":"2 + 2"}'},
            }]}}],
            "usage": {"input_tokens": 8, "output_tokens": 4, "total_tokens": 12},
        }


def test_trace_captures_per_turn_and_total_token_usage():
    result = BenchmarkRunner(UsageAdapter()).run_task(_task())
    assert result.turn.token_usage.prompt_tokens == 8
    assert result.final_turn.token_usage.completion_tokens == 2
    assert result.trace[0].token_usage.total_tokens == 12
    assert result.trace[2].token_usage.total_tokens == 12
    assert result.token_usage.to_dict() == {
        "prompt_tokens": 18,
        "completion_tokens": 6,
        "total_tokens": 24,
    }


class SecretFailureAdapter(ModelAdapter):
    name = "secret-failure"

    def generate(self, messages, tools):
        raise RuntimeError("Bearer extremely-sensitive-secret")


def test_adapter_exception_does_not_leak_details_in_trace_or_report():
    result = BenchmarkRunner(SecretFailureAdapter()).run_task(_task())
    serialized = json.dumps(result.to_dict())
    assert "extremely-sensitive-secret" not in serialized
    assert "RuntimeError" in result.trace[0].error
    assert not result.evaluation.overall_success


def test_legacy_result_without_trace_still_loads():
    result = BenchmarkRunner(MockAgent()).run_task(_task())
    legacy = result.to_dict()
    del legacy["trace"]
    loaded = TaskRunResult.from_dict(legacy)
    assert loaded.trace == []
    assert loaded.final_answer == result.final_answer


def test_serialized_report_redacts_common_credentials():
    task = BenchmarkTask(
        task_id="credentials",
        prompt="Do not expose Bearer hidden-token-value or api_key=hidden-api-value",
        available_tools=[],
        expected_tool=None,
        expected_arguments={},
        expected_answer="anything",
    )
    report = BenchmarkRunner(MockAgent()).run([task])
    serialized = json.dumps(report.to_dict())
    assert "hidden-token-value" not in serialized
    assert "hidden-api-value" not in serialized
    assert "[REDACTED]" in serialized
    RunReport.from_dict(report.to_dict())
