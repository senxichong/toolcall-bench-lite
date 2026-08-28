import json

from toolcall_bench.agent import MockAgent, ModelAdapter
from toolcall_bench.report import render_markdown, write_markdown_report
from toolcall_bench.runner import BenchmarkRunner, load_run_report


class WrongArgumentsAdapter(ModelAdapter):
    name = "wrong-arguments"

    def generate(self, messages, tools):
        if messages[-1]["role"] == "tool":
            return {"content": "5"}
        return {
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "calculator",
                        "arguments": '{"expression":"2 + 3","extra":true}',
                    },
                }
            ]
        }


def test_mock_agent_completes_bundled_benchmark(tmp_path):
    output = tmp_path / "run.json"
    report = BenchmarkRunner(MockAgent()).run("benchmarks/demo.jsonl", output)
    assert report.total_tasks == 25
    assert report.summary()["overall_success"] == {
        "passed": 25,
        "total": 25,
        "rate": 1.0,
    }
    assert load_run_report(output).summary() == report.summary()


def test_runner_records_schema_and_execution_failure():
    report = BenchmarkRunner(WrongArgumentsAdapter()).run([_calculator_task()])
    result = report.results[0]
    assert result.evaluation.tool_selection_correct
    assert not result.evaluation.arguments_correct
    assert not result.evaluation.schema_valid
    assert not result.evaluation.execution_success
    assert not result.evaluation.overall_success


def _calculator_task():
    from toolcall_bench.schemas import BenchmarkTask

    return BenchmarkTask(
        task_id="calc",
        prompt="What is 2 + 2?",
        available_tools=["calculator"],
        expected_tool="calculator",
        expected_arguments={"expression": "2 + 2"},
        expected_answer="4",
        answer_match="exact",
    )


def test_markdown_report_contains_computed_metrics(tmp_path):
    report = BenchmarkRunner(MockAgent()).run([_calculator_task()])
    markdown = render_markdown(report)
    assert "Tool Selection Accuracy | 1 | 1 | 100.0%" in markdown
    assert "| calc | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |" in markdown
    output = write_markdown_report(report, tmp_path / "run.md")
    assert output.read_text(encoding="utf-8") == markdown


class BrokenAdapter(ModelAdapter):
    name = "broken"

    def generate(self, messages, tools):
        raise RuntimeError("offline")


def test_adapter_error_is_scored_instead_of_crashing():
    result = BenchmarkRunner(BrokenAdapter()).run([_calculator_task()]).results[0]
    assert not result.evaluation.overall_success
    assert "adapter error" in result.turn.error
    json.dumps(result.to_dict())
