import pytest

from toolcall_bench.tools import (
    DEFAULT_REGISTRY,
    ToolRegistryError,
    calculator,
    convert_temperature,
    get_word_length,
    lookup_product,
    search_mock_database,
)


def test_default_registry_contains_five_tools():
    assert DEFAULT_REGISTRY.names() == [
        "calculator",
        "get_word_length",
        "convert_temperature",
        "lookup_product",
        "search_mock_database",
    ]


@pytest.mark.parametrize(
    ("expression", "expected"),
    [("25 * 16", 400), ("(18 + 6) / 3", 8), ("7.5 * 4 + 2", 32), ("2 ** 10", 1024)],
)
def test_calculator(expression, expected):
    assert calculator(expression) == expected


@pytest.mark.parametrize(
    "expression",
    ["__import__('os').system('echo unsafe')", "2 ** 100", "[1, 2]", "True + 1"],
)
def test_calculator_rejects_unsafe_or_unbounded_input(expression):
    with pytest.raises(ValueError):
        calculator(expression)


def test_registry_turns_tool_exception_into_data():
    result = DEFAULT_REGISTRY.execute("calculator", {"expression": "1 / 0"})
    assert result.success is False
    assert "ZeroDivisionError" in result.error


def test_schema_rejects_missing_wrong_and_extra_arguments():
    assert DEFAULT_REGISTRY.validate_arguments("calculator", {}) == [
        "arguments.expression is required"
    ]
    assert DEFAULT_REGISTRY.validate_arguments("calculator", {"expression": 4})
    assert DEFAULT_REGISTRY.validate_arguments(
        "calculator", {"expression": "2 + 2", "precision": 2}
    ) == ["arguments.precision is not allowed"]


def test_unknown_tool_lookup_is_explicit():
    with pytest.raises(ToolRegistryError, match="unknown tool"):
        DEFAULT_REGISTRY.get("missing")
    result = DEFAULT_REGISTRY.execute("missing", {})
    assert not result.success


def test_word_length_counts_unicode_code_points():
    assert get_word_length("tool-calling") == 12
    assert get_word_length("工具") == 2


@pytest.mark.parametrize(
    ("value", "source", "target", "expected"),
    [
        (0, "celsius", "fahrenheit", 32),
        (212, "fahrenheit", "celsius", 100),
        (-40, "celsius", "fahrenheit", -40),
        (273.15, "kelvin", "celsius", 0),
    ],
)
def test_temperature_conversion(value, source, target, expected):
    assert convert_temperature(value, source, target) == expected


def test_temperature_rejects_negative_kelvin():
    with pytest.raises(ValueError, match="Kelvin"):
        convert_temperature(-1, "kelvin", "celsius")


def test_product_lookup_is_deterministic_and_returns_copy():
    first = lookup_product("P100")
    assert first["name"] == "Wireless Mouse"
    first["name"] = "changed"
    assert lookup_product("p100")["name"] == "Wireless Mouse"
    with pytest.raises(ValueError, match="not found"):
        lookup_product("p999")


def test_mock_database_search_and_limit():
    assert search_mock_database("Python") == [
        {"id": 1, "topic": "Python", "fact": "Python was created by Guido van Rossum."}
    ]
    assert search_mock_database("i", category="technology", limit=1)[0]["id"] == 1
    assert search_mock_database("absent") == []

