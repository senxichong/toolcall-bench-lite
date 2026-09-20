"""Backend contract tests never download a model or make a network call."""

import io
import json
import urllib.error

import pytest

from toolcall_bench.agent import (
    LLMBackend,
    LocalModelBackend,
    MockBackend,
    ModelAdapterError,
    OpenAICompatibleAdapter,
)
from toolcall_bench.schemas import ToolSpec


def _spec():
    return ToolSpec(
        name="calculator",
        description="Calculate",
        parameters={
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    )


def test_mock_implements_llm_backend():
    assert isinstance(MockBackend(), LLMBackend)


def test_local_backend_with_injected_generator_does_not_load_transformers():
    prompts = []

    def generate(prompt, *, max_new_tokens, do_sample):
        prompts.append(prompt)
        assert max_new_tokens == 32
        assert do_sample is False
        return [{"generated_text": prompt + '{"content":"ready"}'}]

    backend = LocalModelBackend("offline-model", generator=generate, max_new_tokens=32)
    assert backend.generate([{"role": "user", "content": "hello"}], [_spec()]) == {
        "content": "ready"
    }
    assert "calculator" in prompts[0]
    assert "hello" in prompts[0]


def test_local_backend_returns_tool_call_dict_from_generator():
    def generate(prompt, **kwargs):
        return [{"generated_text": '{"tool_calls":[{"function":{"name":"calculator",'
                '"arguments":{"expression":"1+1"}}}]}'}]

    backend = LocalModelBackend("offline-model", generator=generate)
    response = backend.generate([{"role": "user", "content": "1+1"}], [_spec()])
    assert response["tool_calls"][0]["function"]["name"] == "calculator"


def test_local_backend_accepts_chat_style_pipeline_output():
    def generate(prompt, **kwargs):
        return [{"generated_text": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": '{"content":"chat answer"}'},
        ]}]

    backend = LocalModelBackend("offline-model", generator=generate)
    assert backend.generate([{"role": "user", "content": "hello"}], []) == {
        "content": "chat answer"
    }


def test_openai_compatible_backend_sends_function_schema_and_returns_usage(monkeypatch):
    seen = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps({
                "choices": [{"message": {"content": "done"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
            }).encode()

    def fake_urlopen(request, *, timeout):
        seen["url"] = request.full_url
        seen["body"] = json.loads(request.data)
        seen["authorization"] = request.get_header("Authorization")
        seen["timeout"] = timeout
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    backend = OpenAICompatibleAdapter(
        model="demo", base_url="https://example.invalid/v1", api_key="test-key", timeout=3
    )
    response = backend.generate([{"role": "user", "content": "hi"}], [_spec()])
    assert seen["url"] == "https://example.invalid/v1/chat/completions"
    assert seen["body"]["tools"][0]["function"]["name"] == "calculator"
    assert seen["authorization"] == "Bearer test-key"
    assert seen["timeout"] == 3
    assert response["usage"]["total_tokens"] == 6


def test_openai_http_error_does_not_echo_provider_body(monkeypatch):
    def fake_urlopen(request, *, timeout):
        raise urllib.error.HTTPError(
            request.full_url, 429, "rate limit", {}, io.BytesIO(b"api_key=private-secret")
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    backend = OpenAICompatibleAdapter(
        model="demo", base_url="https://example.invalid/v1", api_key="private-secret"
    )
    with pytest.raises(ModelAdapterError, match="HTTP 429") as error:
        backend.generate([{"role": "user", "content": "hi"}], [])
    assert "private-secret" not in str(error.value)


@pytest.mark.parametrize("base_url", ["ftp://example.com", "https://u:p@example.com", "https://example.com?key=x"])
def test_openai_backend_rejects_unsafe_base_url(base_url):
    backend = OpenAICompatibleAdapter(model="demo", base_url=base_url, api_key="secret")
    with pytest.raises(ModelAdapterError, match="Model base URL"):
        backend.generate([{"role": "user", "content": "hi"}], [])
