"""ToolCall-Bench-Lite public package API."""

from .agent import MockAgent, ModelAdapter, OpenAICompatibleAdapter
from .evaluator import Evaluator
from .runner import BenchmarkRunner, run_benchmark
from .schemas import BenchmarkTask, RunReport, ToolCall, ToolSpec
from .tools import DEFAULT_REGISTRY, ToolRegistry

__version__ = "0.1.0"

__all__ = [
    "BenchmarkRunner",
    "BenchmarkTask",
    "DEFAULT_REGISTRY",
    "Evaluator",
    "MockAgent",
    "ModelAdapter",
    "OpenAICompatibleAdapter",
    "RunReport",
    "ToolCall",
    "ToolRegistry",
    "ToolSpec",
    "run_benchmark",
]

