import json

import pytest

from toolcall_bench.datasets import DatasetError, DatasetManager
from toolcall_bench.tasks import load_tasks


def test_import_export_and_idempotent_version(tmp_path):
    manager = DatasetManager(tmp_path / "store")
    first = manager.import_file("demo", "benchmarks/demo.jsonl", metadata={"owner": "test"})
    repeated = manager.import_file("demo", "benchmarks/demo.jsonl")

    assert first.version == "v000001"
    assert repeated == first
    assert first.task_count == 25
    assert manager.list_datasets() == ["demo"]
    assert manager.load_version("demo") == load_tasks("benchmarks/demo.jsonl")

    jsonl_output = manager.export_version("demo", tmp_path / "out.jsonl")
    json_output = manager.export_version("demo", tmp_path / "out.json", format="json")
    assert len(load_tasks(jsonl_output)) == 25
    assert len(json.loads(json_output.read_text(encoding="utf-8"))) == 25


def test_new_version_and_diff(tmp_path):
    manager = DatasetManager(tmp_path / "store")
    original = load_tasks("benchmarks/demo.jsonl")
    manager.import_tasks("demo", original)
    second = manager.import_tasks("demo", original[:-1])

    assert second.version == "v000002"
    differences = manager.diff_versions("demo", "v000001", "v000002")
    assert differences["removed"] == [original[-1].task_id]
    assert differences["added"] == []


def test_version_integrity_detects_tampering(tmp_path):
    manager = DatasetManager(tmp_path / "store")
    version = manager.import_file("demo", "benchmarks/demo.jsonl")
    snapshot = manager.path_for("demo", version.version)
    snapshot.write_text(snapshot.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    # Blank lines are semantically insignificant, so canonical content remains valid.
    assert len(manager.load_version("demo", version.version)) == 25
    snapshot.write_text("{}\n", encoding="utf-8")

    with pytest.raises(DatasetError, match="corrupt dataset version"):
        manager.load_version("demo", version.version)


@pytest.mark.parametrize("name", ["../escape", "bad/name", "two..dots", ""])
def test_dataset_name_rejects_traversal(tmp_path, name):
    manager = DatasetManager(tmp_path / "store")
    with pytest.raises(DatasetError):
        manager.import_file(name, "benchmarks/demo.jsonl")


def test_import_json_array_and_reject_duplicates(tmp_path):
    manager = DatasetManager(tmp_path / "store")
    original = load_tasks("benchmarks/demo.jsonl")
    json_path = tmp_path / "tasks.json"
    json_path.write_text(json.dumps([item.to_dict() for item in original]), encoding="utf-8")

    manifest = manager.import_file("json-demo", json_path)
    assert manifest.task_count == 25
    with pytest.raises(DatasetError, match="duplicate task_id"):
        manager.import_tasks("bad", [original[0], original[0]])

