"""Register a project-specific deterministic tool."""

from toolcall_bench import ToolRegistry, ToolSpec


def reverse_text(text: str) -> str:
    return text[::-1]


registry = ToolRegistry().add(
    ToolSpec(
        name="reverse_text",
        description="Reverse a string.",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
    ),
    reverse_text,
)

print(registry.execute("reverse_text", {"text": "tools"}).value)

