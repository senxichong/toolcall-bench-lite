"""Read and write deterministic JSONL benchmark tasks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .schemas import BenchmarkTask, SchemaValidationError


class TaskFileError(ValueError):
    """Raised when a benchmark file cannot be parsed or validated."""


def load_tasks(path: str | Path) -> list[BenchmarkTask]:
    """Load a UTF-8 JSONL file with line-numbered validation errors."""

    source = Path(path)
    tasks: list[BenchmarkTask] = []
    seen_ids: set[str] = set()
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise TaskFileError(f"cannot read benchmark {source}: {exc}") from exc

    for line_number, line in enumerate(lines, start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TaskFileError(
                f"{source}:{line_number}: invalid JSON: {exc.msg}"
            ) from exc
        try:
            task = BenchmarkTask.from_dict(data)
        except (SchemaValidationError, TypeError) as exc:
            raise TaskFileError(f"{source}:{line_number}: {exc}") from exc
        if task.task_id in seen_ids:
            raise TaskFileError(
                f"{source}:{line_number}: duplicate task_id: {task.task_id}"
            )
        seen_ids.add(task.task_id)
        tasks.append(task)
    if not tasks:
        raise TaskFileError(f"benchmark contains no tasks: {source}")
    return tasks


def save_tasks(tasks: Iterable[BenchmarkTask], path: str | Path) -> Path:
    """Write tasks as compact UTF-8 JSONL and return the destination path."""

    destination = Path(path)
    materialized = list(tasks)
    if not materialized:
        raise TaskFileError("cannot save an empty benchmark")
    task_ids = [task.task_id for task in materialized]
    if len(task_ids) != len(set(task_ids)):
        raise TaskFileError("cannot save duplicate task_id values")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(
        json.dumps(task.to_dict(), ensure_ascii=False, separators=(",", ":"))
        for task in materialized
    )
    destination.write_text(payload + "\n", encoding="utf-8")
    return destination

