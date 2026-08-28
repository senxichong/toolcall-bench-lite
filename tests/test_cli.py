import json

from toolcall_bench.cli import main


def test_run_and_report_commands(tmp_path, capsys):
    output = tmp_path / "latest.json"
    markdown = tmp_path / "custom.md"
    assert main(["run", "benchmarks/demo.jsonl", "--output", str(output)]) == 0
    assert output.exists()
    assert output.with_suffix(".md").exists()
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["summary"]["overall_success"]["passed"] == 25

    assert main(["report", str(output), "--output", str(markdown)]) == 0
    assert markdown.exists()
    captured = capsys.readouterr().out
    assert "Overall success: 25/25" in captured


def test_cli_returns_error_for_missing_benchmark(capsys):
    assert main(["run", "missing.jsonl"]) == 2
    assert "cannot read benchmark" in capsys.readouterr().err

