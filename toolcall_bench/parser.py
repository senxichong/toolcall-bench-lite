"""Normalize provider-native assistant messages without masking malformed output."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from .schemas import AgentTurn, SchemaValidationError, ToolCall


def parse_agent_response(raw: Any) -> AgentTurn:
    """Parse direct, OpenAI-compatible, or legacy function-call output.

    Invalid output becomes an ``AgentTurn(schema_valid=False)`` so a benchmark run
    can continue and score the failure instead of crashing.
    """

    if isinstance(raw, str):
        return AgentTurn(content=raw, raw=raw)
    if not isinstance(raw, Mapping):
        return _invalid(raw, "response must be a string or object")

    message, envelope_error = _unwrap_message(raw)
    if message is None:
        return _invalid(raw, envelope_error or "response has no assistant message")

    errors: list[str] = []
    if envelope_error:
        errors.append(envelope_error)
    content = message.get("content")
    if content is not None and not isinstance(content, str):
        errors.append("content must be a string or null")
        content = None

    call_data: Any = None
    tool_calls = message.get("tool_calls")
    if tool_calls is not None:
        if not isinstance(tool_calls, list):
            errors.append("tool_calls must be a list")
        elif len(tool_calls) != 1:
            errors.append("lite runner expects exactly one tool call per turn")
            if tool_calls:
                call_data = tool_calls[0]
        else:
            call_data = tool_calls[0]
    elif message.get("function_call") is not None:
        call_data = {"function": message["function_call"]}

    tool_call: ToolCall | None = None
    if call_data is not None:
        tool_call, call_errors = _parse_tool_call(call_data)
        errors.extend(call_errors)

    if content is None and tool_call is None:
        errors.append("assistant message contains neither content nor a valid tool call")
    error = "; ".join(dict.fromkeys(errors)) or None
    return AgentTurn(
        content=content,
        tool_call=tool_call,
        raw=raw,
        schema_valid=not errors,
        error=error,
    )


def _unwrap_message(raw: Mapping[str, Any]) -> tuple[Mapping[str, Any] | None, str | None]:
    if "choices" in raw:
        choices = raw.get("choices")
        if not isinstance(choices, list) or not choices:
            return None, "choices must be a non-empty list"
        first = choices[0]
        if not isinstance(first, Mapping) or not isinstance(first.get("message"), Mapping):
            return None, "choices[0].message must be an object"
        error = None if len(choices) == 1 else "response must contain exactly one choice"
        return first["message"], error
    if "message" in raw:
        message = raw.get("message")
        if not isinstance(message, Mapping):
            return None, "message must be an object"
        return message, None
    return raw, None


def _parse_tool_call(raw_call: Any) -> tuple[ToolCall | None, list[str]]:
    if not isinstance(raw_call, Mapping):
        return None, ["tool call must be an object"]
    call_type = raw_call.get("type")
    errors: list[str] = []
    if call_type is not None and call_type != "function":
        errors.append("tool call type must be function")

    function = raw_call.get("function", raw_call)
    if not isinstance(function, Mapping):
        return None, errors + ["tool call function must be an object"]
    name = function.get("name")
    if not isinstance(name, str) or not name.strip():
        return None, errors + ["tool call name must be a non-empty string"]

    arguments_raw = function.get("arguments")
    arguments: dict[str, Any]
    if isinstance(arguments_raw, str):
        try:
            decoded = json.loads(arguments_raw)
        except json.JSONDecodeError:
            decoded = {}
            errors.append("tool call arguments are not valid JSON")
        if isinstance(decoded, Mapping):
            arguments = dict(decoded)
        else:
            arguments = {}
            errors.append("tool call arguments must decode to an object")
    elif isinstance(arguments_raw, Mapping):
        arguments = dict(arguments_raw)
    else:
        arguments = {}
        errors.append("tool call arguments must be a JSON string or object")

    call_id = raw_call.get("id")
    if call_id is not None and not isinstance(call_id, str):
        errors.append("tool call id must be a string")
        call_id = None
    try:
        call = ToolCall(name=name, arguments=arguments, id=call_id)
    except SchemaValidationError as exc:
        return None, errors + [str(exc)]
    return call, errors


def _invalid(raw: Any, message: str) -> AgentTurn:
    return AgentTurn(raw=raw, schema_valid=False, error=message)


parse_response = parse_agent_response

