"""Run the bundled benchmark without API keys or network access."""

from toolcall_bench import BenchmarkRunner, MockAgent
from toolcall_bench.report import write_markdown_report

report = BenchmarkRunner(MockAgent()).run("benchmarks/demo.jsonl", "runs/example.json")
write_markdown_report(report, "runs/example.md")
print(report.summary())

