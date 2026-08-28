"""Deterministic local tools and their lightweight registry."""

from __future__ import annotations

import ast
import math
import operator
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .schemas import ToolExecutionResult, ToolSpec

ToolFunction = Callable[..., Any]


class ToolRegistryError(ValueError):
    """Raised for invalid registration or lookup operations."""


class ToolArgumentError(ValueError):
    """Raised when arguments do not satisfy a tool's JSON Schema."""


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """A schema paired with its local Python implementation."""

    spec: ToolSpec
    function: ToolFunction


class ToolRegistry:
    """Minimal registry that validates and executes deterministic functions."""

    def __init__(self) -> None:
        self._definitions: dict[str, ToolDefinition] = {}

    def register(self, spec: ToolSpec, function: ToolFunction) -> None:
        if spec.name in self._definitions:
            raise ToolRegistryError(f"tool already registered: {spec.name}")
        self._definitions[spec.name] = ToolDefinition(spec=spec, function=function)

    def add(self, spec: ToolSpec, function: ToolFunction) -> ToolRegistry:
        """Register a tool and return the registry for concise examples."""

        self.register(spec, function)
        return self

    def get(self, name: str) -> ToolDefinition:
        try:
            return self._definitions[name]
        except KeyError as exc:
            raise ToolRegistryError(f"unknown tool: {name}") from exc

    def names(self) -> list[str]:
        return list(self._definitions)

    def specs(self, names: list[str] | None = None) -> list[ToolSpec]:
        selected = self.names() if names is None else names
        return [self.get(name).spec for name in selected]

    def validate_arguments(self, name: str, arguments: Mapping[str, Any]) -> list[str]:
        definition = self.get(name)
        return validate_json_schema(arguments, definition.spec.parameters)

    def execute(self, name: str, arguments: Mapping[str, Any]) -> ToolExecutionResult:
        args = dict(arguments)
        try:
            definition = self.get(name)
        except ToolRegistryError as exc:
            return ToolExecutionResult(
                success=False,
                error=str(exc),
                tool_name=name,
                arguments=args,
            )

        errors = validate_json_schema(args, definition.spec.parameters)
        if errors:
            return ToolExecutionResult(
                success=False,
                error="; ".join(errors),
                tool_name=name,
                arguments=args,
            )
        try:
            value = definition.function(**args)
        except Exception as exc:  # Tool failures are benchmark data, not runner crashes.
            return ToolExecutionResult(
                success=False,
                error=f"{type(exc).__name__}: {exc}",
                tool_name=name,
                arguments=args,
            )
        return ToolExecutionResult(
            success=True,
            value=value,
            tool_name=name,
            arguments=args,
        )


def validate_json_schema(value: Any, schema: Mapping[str, Any], path: str = "arguments") -> list[str]:
    """Validate the small JSON Schema subset used by this project.

    Supported keywords are intentionally limited to ``type``, ``properties``,
    ``required``, ``additionalProperties``, ``enum``, numeric bounds, string
    lengths, and homogeneous array ``items``.
    """

    errors: list[str] = []
    expected_type = schema.get("type")
    if expected_type and not _matches_json_type(value, expected_type):
        return [f"{path} must be {expected_type}"]

    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path} must be one of {schema['enum']}")

    if expected_type == "object" and isinstance(value, Mapping):
        properties = schema.get("properties", {})
        for required in schema.get("required", []):
            if required not in value:
                errors.append(f"{path}.{required} is required")
        if schema.get("additionalProperties") is False:
            for key in value:
                if key not in properties:
                    errors.append(f"{path}.{key} is not allowed")
        for key, item in value.items():
            if key in properties:
                errors.extend(validate_json_schema(item, properties[key], f"{path}.{key}"))

    if expected_type == "array" and isinstance(value, list) and "items" in schema:
        for index, item in enumerate(value):
            errors.extend(validate_json_schema(item, schema["items"], f"{path}[{index}]"))

    if expected_type == "string" and isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            errors.append(f"{path} is shorter than minLength")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{path} is longer than maxLength")

    if expected_type in {"number", "integer"} and _is_number(value):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path} is below minimum")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path} is above maximum")
    return errors


def _matches_json_type(value: Any, expected: str) -> bool:
    checks = {
        "object": lambda item: isinstance(item, Mapping),
        "array": lambda item: isinstance(item, list),
        "string": lambda item: isinstance(item, str),
        "number": _is_number,
        "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
        "boolean": lambda item: isinstance(item, bool),
        "null": lambda item: item is None,
    }
    check = checks.get(expected)
    return bool(check and check(value))


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


_BINARY_OPERATORS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def calculator(expression: str) -> int | float:
    """Evaluate a bounded arithmetic expression without using ``eval``."""

    if len(expression) > 100:
        raise ValueError("expression is too long")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError("invalid arithmetic expression") from exc
    result = _evaluate_node(tree.body)
    if not math.isfinite(result) or abs(result) > 1e15:
        raise ValueError("result is outside the allowed range")
    if float(result).is_integer():
        return int(result)
    return round(float(result), 12)


def _evaluate_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if not _is_number(node.value):
            raise ValueError("only numeric constants are allowed")
        if abs(node.value) > 1e12:
            raise ValueError("numeric constant is too large")
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_evaluate_node(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate_node(node.left)
        right = _evaluate_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 12:
            raise ValueError("exponent is outside the allowed range")
        return float(_BINARY_OPERATORS[type(node.op)](left, right))
    raise ValueError("unsupported expression; use numbers and arithmetic operators only")


def get_word_length(word: str) -> int:
    """Return the number of Unicode code points in a word or short string."""

    return len(word)


def convert_temperature(value: float, from_unit: str, to_unit: str) -> int | float:
    """Convert a temperature among Celsius, Fahrenheit, and Kelvin."""

    if from_unit == "kelvin" and value < 0:
        raise ValueError("Kelvin cannot be below zero")
    if from_unit == "celsius":
        celsius = float(value)
    elif from_unit == "fahrenheit":
        celsius = (float(value) - 32) * 5 / 9
    elif from_unit == "kelvin":
        celsius = float(value) - 273.15
    else:
        raise ValueError(f"unsupported source unit: {from_unit}")

    if to_unit == "celsius":
        result = celsius
    elif to_unit == "fahrenheit":
        result = celsius * 9 / 5 + 32
    elif to_unit == "kelvin":
        result = celsius + 273.15
    else:
        raise ValueError(f"unsupported target unit: {to_unit}")
    if result < 0 and to_unit == "kelvin":
        raise ValueError("conversion would be below absolute zero")
    rounded = round(result, 10)
    return int(rounded) if rounded.is_integer() else rounded


_PRODUCTS: dict[str, dict[str, Any]] = {
    "p100": {
        "product_id": "p100",
        "name": "Wireless Mouse",
        "price": 25.99,
        "in_stock": True,
    },
    "p200": {
        "product_id": "p200",
        "name": "Mechanical Keyboard",
        "price": 79.5,
        "in_stock": False,
    },
    "p300": {
        "product_id": "p300",
        "name": "USB-C Hub",
        "price": 39.0,
        "in_stock": True,
    },
}


def lookup_product(product_id: str) -> dict[str, Any]:
    """Return a product from a deterministic in-memory catalog."""

    normalized = product_id.lower()
    if normalized not in _PRODUCTS:
        raise ValueError(f"product not found: {product_id}")
    return dict(_PRODUCTS[normalized])


_MOCK_RECORDS: tuple[tuple[str, dict[str, Any]], ...] = (
    (
        "technology",
        {"id": 1, "topic": "Python", "fact": "Python was created by Guido van Rossum."},
    ),
    (
        "technology",
        {"id": 2, "topic": "SQLite", "fact": "SQLite is an embedded SQL database."},
    ),
    (
        "science",
        {"id": 3, "topic": "Kelvin", "fact": "Zero kelvin is absolute zero."},
    ),
    (
        "geography",
        {"id": 4, "topic": "Singapore", "fact": "Singapore is a city-state."},
    ),
)


def search_mock_database(
    query: str, category: str | None = None, limit: int = 5
) -> list[dict[str, Any]]:
    """Case-insensitively search a tiny, deterministic record collection."""

    needle = query.casefold().strip()
    if not needle:
        raise ValueError("query must not be blank")
    matches: list[dict[str, Any]] = []
    for record_category, record in _MOCK_RECORDS:
        if category is not None and category.casefold() != record_category.casefold():
            continue
        haystack = " ".join(str(value) for value in record.values()).casefold()
        if needle in haystack:
            matches.append(dict(record))
        if len(matches) >= limit:
            break
    return matches


def build_default_registry() -> ToolRegistry:
    """Create the built-in deterministic registry."""

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="calculator",
            description="Safely evaluate a basic arithmetic expression.",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "minLength": 1, "maxLength": 100}
                },
                "required": ["expression"],
                "additionalProperties": False,
            },
        ),
        calculator,
    )
    registry.register(
        ToolSpec(
            name="get_word_length",
            description="Count characters in a supplied word or short string.",
            parameters={
                "type": "object",
                "properties": {"word": {"type": "string", "minLength": 1}},
                "required": ["word"],
                "additionalProperties": False,
            },
        ),
        get_word_length,
    )
    unit_schema = {"type": "string", "enum": ["celsius", "fahrenheit", "kelvin"]}
    registry.register(
        ToolSpec(
            name="convert_temperature",
            description="Convert a temperature between Celsius, Fahrenheit, and Kelvin.",
            parameters={
                "type": "object",
                "properties": {
                    "value": {"type": "number"},
                    "from_unit": unit_schema,
                    "to_unit": unit_schema,
                },
                "required": ["value", "from_unit", "to_unit"],
                "additionalProperties": False,
            },
        ),
        convert_temperature,
    )
    registry.register(
        ToolSpec(
            name="lookup_product",
            description="Look up one product record by product ID.",
            parameters={
                "type": "object",
                "properties": {"product_id": {"type": "string", "minLength": 1}},
                "required": ["product_id"],
                "additionalProperties": False,
            },
        ),
        lookup_product,
    )
    registry.register(
        ToolSpec(
            name="search_mock_database",
            description="Search deterministic local records; no internet access is used.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "minLength": 1},
                    "category": {"type": "string", "minLength": 1},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 10},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        ),
        search_mock_database,
    )
    return registry


DEFAULT_REGISTRY = build_default_registry()

