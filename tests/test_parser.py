import pytest

from toolcall_bench.parser import parse_agent_response


def test_parse_plain_content():
    turn = parse_agent_response({"content": "Hello!"})
    assert turn.content == "Hello!"
    assert turn.tool_call is None
    assert turn.schema_valid


def test_parse_openai_envelope_and_tool_call():
    raw = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "calculator",
                                "arguments": '{"expression":"2 + 2"}',
                            },
                        }
                    ],
                }
            }
        ]
    }
    turn = parse_agent_response(raw)
    assert turn.schema_valid
    assert turn.tool_call.name == "calculator"
    assert turn.tool_call.arguments == {"expression": "2 + 2"}
    assert turn.tool_call.id == "call_1"


def test_parse_legacy_function_call():
    turn = parse_agent_response(
        {"content": None, "function_call": {"name": "calculator", "arguments": "{}"}}
    )
    assert turn.tool_call.name == "calculator"
    assert turn.schema_valid


def test_invalid_argument_json_keeps_tool_name_for_selection_scoring():
    turn = parse_agent_response(
        {
            "tool_calls": [
                {"type": "function", "function": {"name": "calculator", "arguments": "{"}}
            ]
        }
    )
    assert turn.tool_call.name == "calculator"
    assert turn.tool_call.arguments == {}
    assert not turn.schema_valid
    assert "not valid JSON" in turn.error


def test_multiple_calls_are_invalid_but_first_call_is_preserved():
    call = {"function": {"name": "calculator", "arguments": {"expression": "1 + 1"}}}
    turn = parse_agent_response({"tool_calls": [call, call]})
    assert turn.tool_call.name == "calculator"
    assert not turn.schema_valid
    assert "exactly one" in turn.error


@pytest.mark.parametrize("raw", [None, [], {"choices": []}, {"content": 42}, {}])
def test_malformed_responses_become_invalid_turns(raw):
    turn = parse_agent_response(raw)
    assert not turn.schema_valid
    assert turn.error
