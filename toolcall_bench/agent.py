"""Model adapters used by ToolCall-Bench-Lite.

Adapters return provider-native dictionaries.  Parsing and validation deliberately
live elsewhere so malformed model output can be measured instead of hidden here.
"""

from __future__ import annotations

import ast
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from typing import Any, Callable

from .schemas import ToolSpec


class ModelAdapterError(RuntimeError):
    """Raised when an adapter cannot obtain a model response."""


class LLMBackend(ABC):
    """Provider-neutral interface for one agent model turn."""

    name = "model"

    @abstractmethod
    def generate(
        self, messages: list[dict[str, Any]], tools: list[ToolSpec]
    ) -> dict[str, Any] | str:
        """Generate one assistant turn from messages and available tools."""

    def complete(
        self, messages: list[dict[str, Any]], tools: list[ToolSpec]
    ) -> dict[str, Any] | str:
        """Compatibility alias with a more conversational method name."""

        return self.generate(messages, tools)


class ModelAdapter(LLMBackend):
    """Backward-compatible name for the public LLM backend contract."""


class MockAgent(ModelAdapter):
    """Deterministic rule-based adapter for the bundled demo.

    It only sees prompts, tool schemas, and tool results; it never reads benchmark
    gold fields.  That makes it useful for examples and regression tests, not as a
    claim about real-model quality.
    """

    name = "mock"

    _UNIT_ALIASES = {
        "c": "celsius",
        "°c": "celsius",
        "celsius": "celsius",
        "f": "fahrenheit",
        "°f": "fahrenheit",
        "fahrenheit": "fahrenheit",
        "k": "kelvin",
        "kelvin": "kelvin",
    }

    def generate(
        self, messages: list[dict[str, Any]], tools: list[ToolSpec]
    ) -> dict[str, Any] | str:
        if not messages:
            raise ModelAdapterError("MockAgent requires at least one message")

        if messages[-1].get("role") == "tool":
            return self._answer_from_tool(messages)

        prompt = self._latest_user_prompt(messages)
        available = {tool.name for tool in tools}
        call = self._choose_call(prompt, available)
        if call is not None:
            name, arguments = call
            return {
                "content": None,
                "tool_calls": [
                    {
                        "id": "mock_call_1",
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(arguments, ensure_ascii=False),
                        },
                    }
                ],
            }
        return {"content": self._answer_without_tool(prompt)}

    @staticmethod
    def _latest_user_prompt(messages: list[dict[str, Any]]) -> str:
        for message in reversed(messages):
            if message.get("role") == "user":
                return str(message.get("content", ""))
        return ""

    def _choose_call(
        self, prompt: str, available: set[str]
    ) -> tuple[str, dict[str, Any]] | None:
        lowered = prompt.lower()

        if "convert_temperature" in available and self._mentions_temperature(lowered):
            temperature = self._temperature_arguments(prompt)
            if temperature is not None:
                return "convert_temperature", temperature

        if "get_word_length" in available and re.search(
            r"(?:word|text|string).{0,20}length|(?:length|characters?)|"
            r"(?:how many|count).{0,16}(?:letters|characters?)",
            lowered,
        ):
            text = self._quoted_text(prompt)
            if text is not None:
                return "get_word_length", {"word": text}

        if "lookup_product" in available:
            product = re.search(r"\b[pP]\d{3}\b", prompt)
            if product and any(word in lowered for word in ("product", "price", "cost", "lookup")):
                return "lookup_product", {"product_id": product.group(0).lower()}

        if "search_mock_database" in available and any(
            word in lowered for word in ("search", "database", "find records", "find entries")
        ):
            query = self._quoted_text(prompt)
            if query is None:
                match = re.search(
                    r"(?:for|about|matching)\s+([\w .+-]+?)(?:\s+in\s+the\s+\w+\s+category)?[?.!]?$",
                    prompt,
                    re.IGNORECASE,
                )
                query = match.group(1).strip() if match else ""
            arguments: dict[str, Any] = {"query": query}
            category = re.search(r"category\s+['\"]?([\w-]+)", prompt, re.IGNORECASE)
            if category:
                arguments["category"] = category.group(1).lower()
            limit = re.search(r"(?:top|limit(?:ed)?\s+to)\s+(\d+)", lowered)
            if limit:
                arguments["limit"] = int(limit.group(1))
            return "search_mock_database", arguments

        if "calculator" in available:
            expression = self._arithmetic_expression(prompt)
            if expression is not None:
                return "calculator", {"expression": expression}

        return None

    @staticmethod
    def _mentions_temperature(prompt: str) -> bool:
        return "convert" in prompt and any(
            unit in prompt for unit in ("celsius", "fahrenheit", "kelvin", "°c", "°f")
        )

    def _temperature_arguments(self, prompt: str) -> dict[str, Any] | None:
        lowered = prompt.lower()
        value_match = re.search(
            r"(-?\d+(?:\.\d+)?)\s*(?:degrees?\s*)?°?\s*"
            r"(celsius|fahrenheit|kelvin|[cfk])\b",
            lowered,
        )
        target_match = re.search(
            r"(?:to|into)\s+(?:degrees?\s+)?(celsius|fahrenheit|kelvin|[cfk])\b", lowered
        )
        if not value_match or not target_match:
            return None
        return {
            "value": float(value_match.group(1)),
            "from_unit": self._UNIT_ALIASES[value_match.group(2)],
            "to_unit": self._UNIT_ALIASES[target_match.group(1)],
        }

    @staticmethod
    def _quoted_text(prompt: str) -> str | None:
        match = re.search(r"[\"']([^\"']+)[\"']", prompt)
        return match.group(1) if match else None

    @staticmethod
    def _arithmetic_expression(prompt: str) -> str | None:
        # Prefer explicitly quoted expressions, then the longest operator-bearing span.
        quoted = MockAgent._quoted_text(prompt)
        if quoted and re.fullmatch(r"[\d\s+\-*/().%^]+", quoted):
            return quoted.strip().replace("^", "**")
        candidates = re.findall(r"[\d\s+\-*/%^().]+", prompt)
        valid: list[str] = []
        for candidate in candidates:
            normalized = candidate.strip().rstrip(".").strip().replace("^", "**")
            if not re.search(r"[+\-*/%]", normalized) or not re.search(r"\d", normalized):
                continue
            try:
                ast.parse(normalized, mode="eval")
            except SyntaxError:
                continue
            valid.append(normalized)
        return max(valid, key=len) if valid else None

    def _answer_from_tool(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        prompt = self._latest_user_prompt(messages)
        payload = messages[-1].get("content")
        if isinstance(payload, str):
            try:
                result = json.loads(payload)
            except json.JSONDecodeError:
                result = {"success": False, "error": payload}
        elif isinstance(payload, dict):
            result = payload
        else:
            result = {"success": False, "error": "Invalid tool result"}

        if not result.get("success", False):
            return {"content": f"Tool error: {result.get('error', 'unknown error')}"}

        value = result.get("value")
        tool_name = str(result.get("name", messages[-1].get("name", "")))
        lowered = prompt.lower()
        if "json" in lowered:
            answer = json.dumps(value, ensure_ascii=False, sort_keys=True)
        elif tool_name == "lookup_product" and isinstance(value, dict):
            if "price" in lowered or "cost" in lowered:
                answer = self._format_scalar(value.get("price"))
            elif "name" in lowered:
                answer = str(value.get("name", ""))
            else:
                answer = json.dumps(value, ensure_ascii=False, sort_keys=True)
        elif tool_name == "search_mock_database" and isinstance(value, list):
            answer = json.dumps(value, ensure_ascii=False, sort_keys=True)
        else:
            answer = self._format_scalar(value)
        return {"content": answer}

    @staticmethod
    def _format_scalar(value: Any) -> str:
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        return str(value)

    @staticmethod
    def _answer_without_tool(prompt: str) -> str:
        lowered = prompt.strip().lower()
        if lowered in {"say hello.", "say hello", "hello", "hi"}:
            return "Hello!"
        if "clear" in lowered and "sky" in lowered and "color" in lowered:
            return "Blue."
        if "length" in lowered and any(
            phrase in lowered for phrase in ("not provided", "no word", "forgot the word")
        ):
            return "Please specify the word to measure."
        if "product" in lowered and any(
            phrase in lowered for phrase in ("not provided", "no product id", "forgot the product")
        ):
            return "Please specify the product ID."
        if "convert" in lowered and any(
            unit in lowered for unit in ("celsius", "fahrenheit", "kelvin", "°c", "°f")
        ):
            return "Please specify the target temperature unit."
        json_match = re.search(r"(\{.*\})", prompt)
        if "without calling a tool" in lowered and json_match:
            try:
                value = json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
            else:
                return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        exact_match = re.search(r"reply exactly:\s*(.+)$", prompt, re.IGNORECASE)
        if exact_match:
            return exact_match.group(1).strip()
        return "I can answer this without calling a tool."


class OpenAICompatibleAdapter(ModelAdapter):
    """Adapter for an OpenAI-compatible ``/chat/completions`` endpoint.

    Configuration defaults to environment variables, so importing the package never
    requires a key or a network dependency:

    - ``TOOLCALL_BENCH_MODEL``
    - ``TOOLCALL_BENCH_BASE_URL``
    - ``TOOLCALL_BENCH_API_KEY``
    """

    name = "openai-compatible"

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.model = model or os.getenv("TOOLCALL_BENCH_MODEL", "")
        self.base_url = base_url or os.getenv("TOOLCALL_BENCH_BASE_URL", "")
        self.api_key = api_key or os.getenv("TOOLCALL_BENCH_API_KEY", "")
        self.timeout = timeout

    def generate(
        self, messages: list[dict[str, Any]], tools: list[ToolSpec]
    ) -> dict[str, Any]:
        if not self.model:
            raise ModelAdapterError("Set TOOLCALL_BENCH_MODEL before using this adapter")
        if not self.base_url:
            raise ModelAdapterError("Set TOOLCALL_BENCH_BASE_URL before using this adapter")
        if not self.api_key:
            raise ModelAdapterError("Set TOOLCALL_BENCH_API_KEY before using this adapter")

        parsed_url = urllib.parse.urlsplit(self.base_url)
        if (
            parsed_url.scheme not in {"http", "https"}
            or not parsed_url.netloc
            or parsed_url.username is not None
            or parsed_url.password is not None
            or parsed_url.query
            or parsed_url.fragment
        ):
            raise ModelAdapterError("Model base URL must be an HTTP(S) URL without credentials")
        endpoint = self.base_url.rstrip("/")
        if not endpoint.endswith("/chat/completions"):
            endpoint += "/chat/completions"
        body = {
            "model": self.model,
            "messages": messages,
            "tools": [self._openai_tool(tool) for tool in tools],
            "tool_choice": "auto",
            "temperature": 0,
        }
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            # Provider bodies commonly echo Authorization or request data. Never
            # copy them into exceptions, traces, or persisted reports.
            raise ModelAdapterError(f"Model endpoint returned HTTP {exc.code}") from None
        except (urllib.error.URLError, TimeoutError):
            raise ModelAdapterError("Could not reach model endpoint") from None

        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            raise ModelAdapterError("Model endpoint returned invalid JSON") from None
        if not isinstance(parsed, dict):
            raise ModelAdapterError("Model endpoint response must be a JSON object")
        return parsed

    @staticmethod
    def _openai_tool(spec: ToolSpec) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters,
            },
        }


class LocalModelBackend(ModelAdapter):
    """Lazy local text-generation backend using optional ``transformers``.

    A compatible callable may be injected for tests or an existing inference
    runtime. It receives a JSON-formatted prompt and generation keyword arguments.
    No model is loaded until the first ``generate`` call.
    """

    name = "local"

    def __init__(
        self,
        model: str,
        generator: Callable[..., Any] | None = None,
        max_new_tokens: int = 512,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("local model must be a non-empty string")
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        self.model = model
        self._generator = generator
        self.max_new_tokens = max_new_tokens

    def _load_generator(self) -> Callable[..., Any]:
        if self._generator is None:
            try:
                from transformers import pipeline
            except ImportError:
                raise ModelAdapterError(
                    "Local inference requires the optional transformers package"
                ) from None
            try:
                self._generator = pipeline(
                    task="text-generation", model=self.model, trust_remote_code=False
                )
            except Exception:
                # Model paths, registry credentials, and remote error bodies are
                # intentionally omitted from persisted benchmark failures.
                raise ModelAdapterError("Could not initialize local model") from None
        return self._generator

    def generate(
        self, messages: list[dict[str, Any]], tools: list[ToolSpec]
    ) -> dict[str, Any] | str:
        prompt = (
            "You are a tool-calling assistant. Return one JSON object with either "
            "{\"content\": \"answer\"} or {\"tool_calls\": [{\"type\": \"function\", "
            "\"function\": {\"name\": \"tool_name\", \"arguments\": {}}}]}. "
            "Use at most one tool call. Available tools and conversation follow.\n"
            + json.dumps(
                {"tools": [tool.to_openai_tool() for tool in tools], "messages": messages},
                ensure_ascii=False,
            )
            + "\nAssistant JSON:"
        )
        try:
            generated = self._load_generator()(
                prompt, max_new_tokens=self.max_new_tokens, do_sample=False
            )
        except ModelAdapterError:
            raise
        except Exception:
            raise ModelAdapterError("Local model generation failed") from None
        if isinstance(generated, list):
            generated = generated[0] if generated else None
        if isinstance(generated, dict) and "generated_text" in generated:
            generated = generated["generated_text"]
        if isinstance(generated, list) and generated and isinstance(generated[-1], dict):
            generated = generated[-1]
        if isinstance(generated, dict) and generated.get("role") == "assistant":
            generated = generated.get("content")
        if isinstance(generated, dict):
            return generated
        if not isinstance(generated, str):
            raise ModelAdapterError("Local model returned an unsupported response")
        completion = generated[len(prompt):] if generated.startswith(prompt) else generated
        completion = completion.strip()
        try:
            decoded = json.loads(completion)
        except json.JSONDecodeError:
            return {"content": completion}
        return decoded if isinstance(decoded, dict) else {"content": completion}


# A convenient alias used by early examples of the project.
MockAdapter = MockAgent
MockBackend = MockAgent
LocalModelAdapter = LocalModelBackend
