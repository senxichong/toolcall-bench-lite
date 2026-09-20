import json

from toolcall_bench.evaluator import Evaluator, answer_matches, arguments_match
from toolcall_bench.parser import parse_agent_response
from toolcall_bench.schemas import BenchmarkTask, ToolExecutionResult


def make_task(**overrides):
    values = {
        "task_id": "calc",
        "prompt": "What is 2 + 2?",
        "available_tools": ["calculator", "get_word_length"],
        "expected_tool": "calculator",
        "expected_arguments": {"expression": "2 + 2"},
        "expected_answer": "4",
        "answer_match": "exact",
    }
    values.update(overrides)
    return BenchmarkTask(**values)


def tool_turn(name="calculator", arguments=None):
    if arguments is None:
        arguments = {"expression": "2 + 2"}
    return parse_agent_response(
        {
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments),
                    },
                }
            ]
        }
    )


def test_all_metrics_and_overall_success():
    evaluation = Evaluator().evaluate(
        make_task(),
        tool_turn(),
        ToolExecutionResult(success=True, value=4, tool_name="calculator"),
        parse_agent_response({"content": "4"}),
    )
    assert evaluation.overall_success
    assert evaluation.to_dict()["arguments_correct"] is True


def test_wrong_tool_can_execute_but_overall_fails():
    evaluation = Evaluator().evaluate(
        make_task(),
        tool_turn("get_word_length", {"word": "test"}),
        ToolExecutionResult(success=True, value=4, tool_name="get_word_length"),
        parse_agent_response({"content": "4"}),
    )
    assert not evaluation.tool_selection_correct
    assert not evaluation.arguments_correct
    assert evaluation.schema_valid
    assert evaluation.execution_success
    assert evaluation.final_answer_correct
    assert not evaluation.overall_success


def test_missing_required_argument_fails_schema_and_execution():
    turn = tool_turn(arguments={})
    execution = ToolExecutionResult(success=False, error="missing argument")
    evaluation = Evaluator().evaluate(make_task(), turn, execution)
    assert not evaluation.arguments_correct
    assert not evaluation.schema_valid
    assert not evaluation.execution_success


def test_second_tool_call_cannot_score_as_valid_final_answer():
    final = parse_agent_response(
        {
            "content": "4",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "calculator", "arguments": '{"expression":"2 + 2"}'},
                }
            ],
        }
    )
    evaluation = Evaluator().evaluate(
        make_task(),
        tool_turn(),
        ToolExecutionResult(success=True, value=4, tool_name="calculator"),
        final,
    )
    assert not evaluation.schema_valid
    assert not evaluation.final_answer_correct
    assert not evaluation.overall_success


def test_correct_no_call_scores_as_success():
    task = make_task(
        expected_tool=None,
        expected_arguments={},
        expected_answer="Hello!",
        prompt="Say hello.",
    )
    evaluation = Evaluator().evaluate(task, parse_agent_response({"content": "Hello!"}))
    assert evaluation.overall_success


def test_argument_matching_normalizes_expressions_and_numbers():
    assert arguments_match(
        {"expression": "(2 + 3) * 4", "value": 1},
        {"expression": "(2+3)*4", "value": 1.0},
    )
    assert not arguments_match({"value": 1}, {"value": "1"})
    assert not arguments_match({"a": 1}, {"a": 1, "b": 2})


def test_answer_match_modes():
    assert answer_matches({"ok": True, "n": 2}, '{"n": 2, "ok": true}', "json")
    assert answer_matches(32, "32.0000000001", "numeric")
    assert answer_matches("result is 4", "The result is 4 today", "contains")
    assert answer_matches("Hello World", " hello   world ", "auto")
    assert not answer_matches("Hello!", "hello!", "exact")
