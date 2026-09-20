import json

from toolcall_bench.cli import main


def test_run_and_report_commands(tmp_path, capsys):
    output = tmp_path / "latest.json"
    markdown = tmp_path / "custom.md"
    assert main(["run", "benchmarks/demo.jsonl", "--output", str(output)]) == 0
    assert output.exists()
    assert output.with_suffix(".md").exists()
    assert output.with_suffix(".badcases.json").exists()
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["summary"]["overall_success"]["passed"] == 25

    assert main(["report", str(output), "--output", str(markdown)]) == 0
    assert markdown.exists()
    captured = capsys.readouterr().out
    assert "Overall success: 25/25" in captured


def test_cli_returns_error_for_missing_benchmark(capsys):
    assert main(["run", "missing.jsonl"]) == 2
    assert "cannot read benchmark" in capsys.readouterr().err


def test_cli_dataset_version_lifecycle(tmp_path, capsys):
    root = tmp_path / "datasets"
    prefix = ["dataset", "--root", str(root)]
    assert main(prefix + ["import", "demo", "benchmarks/demo.jsonl"]) == 0
    import_result = json.loads(capsys.readouterr().out)
    assert import_result["version"] == "v000001"
    assert main(prefix + ["verify", "demo"]) == 0
    assert json.loads(capsys.readouterr().out)["verified"] is True
    destination = tmp_path / "export.json"
    assert main(prefix + ["export", "demo", str(destination), "--format", "json"]) == 0
    assert len(json.loads(destination.read_text(encoding="utf-8"))) == 25

    output = tmp_path / "versioned-run.json"
    assert main(
        [
            "run",
            "--dataset-name",
            "demo",
            "--datasets-dir",
            str(root),
            "--output",
            str(output),
        ]
    ) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["metadata"]["dataset_version"] == "v000001"
    assert report["metadata"]["dataset_sha256"] == import_result["sha256"]
