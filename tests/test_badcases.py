import json

from toolcall_bench.badcases import (
    FailureCategory,
    RuleBasedBadcaseAnalyzer,
    analyze_badcases,
    load_badcase_report,
    write_badcase_report,
)
from toolcall_bench.runner import BenchmarkRunner


class WrongToolAdapter:
    name = "wrong-tool"

    def generate(self, messages, tools):
        return {"tool_calls": [{"type": "function", "function": {"name": "wrong", "arguments": "{}"}}]}


def test_mock_demo_has_no_badcases():
    report = BenchmarkRunner().run("benchmarks/demo.jsonl")
    analyzed = analyze_badcases(report)

    assert analyzed.failed_tasks == 0
    assert analyzed.badcases == ()


def test_wrong_tool_is_structured_badcase(tmp_path):
    task = BenchmarkRunner().run("benchmarks/demo.jsonl").results[0].task
    report = BenchmarkRunner(adapter=WrongToolAdapter()).run([task])
    analyzed = RuleBasedBadcaseAnalyzer().analyze(report)
    record = analyzed.badcases[0]

    assert record.primary_category == FailureCategory.WRONG_TOOL
    assert record.expected_tool == task.expected_tool
    assert record.actual_tool == "wrong"
    assert analyzed.category_counts["wrong_tool"] == 1

    output = write_badcase_report(analyzed, tmp_path / "badcases.json")
    loaded = load_badcase_report(output)
    assert loaded["badcases"][0]["primary_category"] == "wrong_tool"
    assert loaded["failed_tasks"] == 1
    assert json.loads(output.read_text(encoding="utf-8"))["schema_version"] == 1
