"""ToolCall-Bench-Lite public package API."""

from .agent import (
    LLMBackend,
    LocalModelBackend,
    MockAgent,
    MockBackend,
    ModelAdapter,
    OpenAICompatibleAdapter,
)
from .badcases import BadcaseReport, FailureCategory, analyze_badcases
from .datasets import DatasetManager, DatasetVersion
from .evaluator import Evaluator
from .runner import BenchmarkRunner, run_benchmark
from .schemas import BenchmarkTask, RunReport, ToolCall, ToolSpec
from .tools import DEFAULT_REGISTRY, ToolRegistry

__version__ = "0.2.0"

__all__ = [
    "BenchmarkRunner",
    "BenchmarkTask",
    "BadcaseReport",
    "DEFAULT_REGISTRY",
    "DatasetManager",
    "DatasetVersion",
    "Evaluator",
    "FailureCategory",
    "LLMBackend",
    "LocalModelBackend",
    "MockAgent",
    "MockBackend",
    "ModelAdapter",
    "OpenAICompatibleAdapter",
    "RunReport",
    "ToolCall",
    "ToolRegistry",
    "ToolSpec",
    "analyze_badcases",
    "run_benchmark",
]
