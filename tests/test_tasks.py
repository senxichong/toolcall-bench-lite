import json

import pytest

from toolcall_bench.schemas import BenchmarkTask
from toolcall_bench.tasks import TaskFileError, load_tasks, save_tasks


def test_bundled_benchmark_has_25_unique_tasks():
    tasks = load_tasks("benchmarks/demo.jsonl")
    assert len(tasks) == 25
    assert len({task.task_id for task in tasks}) == 25
    assert {task.expected_tool for task in tasks} >= {
        None,
        "calculator",
        "get_word_length",
        "convert_temperature",
        "lookup_product",
        "search_mock_database",
    }


def test_task_round_trip(tmp_path):
    task = BenchmarkTask(
        task_id="demo",
        prompt="Say hello.",
        available_tools=[],
        expected_tool=None,
        expected_arguments={},
        expected_answer="Hello!",
        answer_match="exact",
    )
    path = save_tasks([task], tmp_path / "tasks.jsonl")
    assert load_tasks(path) == [task]


def test_duplicate_ids_report_line_number(tmp_path):
    row = {
        "task_id": "same",
        "prompt": "hello",
        "available_tools": [],
        "expected_tool": None,
        "expected_arguments": {},
        "expected_answer": "hello",
    }
    path = tmp_path / "duplicate.jsonl"
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(TaskFileError, match=r":2: duplicate task_id"):
        load_tasks(path)


def test_bad_json_and_unknown_fields_are_rejected(tmp_path):
    bad_json = tmp_path / "bad.jsonl"
    bad_json.write_text("{broken}\n", encoding="utf-8")
    with pytest.raises(TaskFileError, match="invalid JSON"):
        load_tasks(bad_json)

    unknown = tmp_path / "unknown.jsonl"
    unknown.write_text(
        json.dumps(
            {
                "task_id": "x",
                "prompt": "x",
                "available_tools": [],
                "expected_tool": None,
                "expected_arguments": {},
                "expected_answer": "x",
                "surprise": True,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(TaskFileError, match="unknown benchmark task field"):
        load_tasks(unknown)


def test_empty_benchmark_is_rejected(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("\n# only a comment\n", encoding="utf-8")
    with pytest.raises(TaskFileError, match="contains no tasks"):
        load_tasks(path)

