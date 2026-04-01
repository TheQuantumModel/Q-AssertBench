"""Client abstractions for repeated generation trials."""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib import request as urllib_request


@dataclass(frozen=True)
class GenerationResponse:
    """Structured raw model response."""

    text: str
    payload: dict[str, Any] = field(default_factory=dict)


class ModelClient(Protocol):
    """Minimal interface required by the generation driver."""

    @property
    def model_id(self) -> str:
        ...

    def generate(self, *, prompt_text: str, task_id: str, trial_index: int) -> GenerationResponse:
        ...


def _load_openai_sdk():
    return importlib.import_module("openai")


def _extract_message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                    continue
            text = getattr(item, "text", None)
            if isinstance(text, str):
                parts.append(text)
        return "\n".join(part for part in parts if part)
    return ""


@dataclass(frozen=True)
class StaticResponseClient:
    """Local client that returns the same response for every request."""

    model_name: str
    response_text: str

    @property
    def model_id(self) -> str:
        return self.model_name

    def generate(self, *, prompt_text: str, task_id: str, trial_index: int) -> GenerationResponse:
        return GenerationResponse(
            text=self.response_text,
            payload={
                "provider_type": "static",
                "task_id": task_id,
                "trial_index": trial_index,
                "prompt_length": len(prompt_text),
                "temperature": None,
                "max_output_tokens": None,
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
            },
        )


@dataclass(frozen=True)
class OpenAICompatibleClient:
    """Model client backed by an OpenAI SDK compatible chat-completions endpoint."""

    model_name: str
    api_key: str
    api_base_url: str | None = None
    temperature: float = 0.0
    max_output_tokens: int = 2048
    _client: Any = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        try:
            sdk = _load_openai_sdk()
        except ModuleNotFoundError as exc:  # pragma: no cover - exercised via patching in tests
            raise RuntimeError(
                "The openai package is required for OpenAI-compatible generation. "
                "Install it in project_code/.venv first."
            ) from exc

        object.__setattr__(
            self,
            "_client",
            sdk.OpenAI(
                api_key=self.api_key,
                base_url=self.api_base_url,
            ),
        )

    @property
    def model_id(self) -> str:
        return self.model_name

    def _uses_openai_official_gpt5_endpoint(self) -> bool:
        if not self.api_base_url:
            return False
        return "api.openai.com" in self.api_base_url and self.model_name.startswith("gpt-5")

    def generate(self, *, prompt_text: str, task_id: str, trial_index: int) -> GenerationResponse:
        request_kwargs = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt_text}],
            "temperature": self.temperature,
        }
        if self._uses_openai_official_gpt5_endpoint():
            request_kwargs["max_completion_tokens"] = self.max_output_tokens
        else:
            request_kwargs["max_tokens"] = self.max_output_tokens

        response = self._client.chat.completions.create(**request_kwargs)
        choices = getattr(response, "choices", None) or []
        choice = choices[0] if choices else None
        message = getattr(choice, "message", None)
        content = getattr(message, "content", "")
        text = _extract_message_text(content)
        usage = getattr(response, "usage", None)

        return GenerationResponse(
            text=text,
            payload={
                "provider_type": "openai-compatible",
                "task_id": task_id,
                "trial_index": trial_index,
                "api_base_url": self.api_base_url,
                "model_name": self.model_name,
                "temperature": self.temperature,
                "max_output_tokens": self.max_output_tokens,
                "response_id": getattr(response, "id", None),
                "response_model": getattr(response, "model", None),
                "finish_reason": getattr(choice, "finish_reason", None),
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            },
        )


@dataclass(frozen=True)
class GeminiGenerativeLanguageClient:
    """Model client backed by Google's Gemini Developer API."""

    model_name: str
    api_key: str
    api_base_url: str | None = None
    temperature: float = 0.0
    max_output_tokens: int = 2048

    @property
    def model_id(self) -> str:
        return self.model_name

    def generate(self, *, prompt_text: str, task_id: str, trial_index: int) -> GenerationResponse:
        base_url = (self.api_base_url or "https://generativelanguage.googleapis.com").rstrip("/")
        url = f"{base_url}/v1beta/models/{self.model_name}:generateContent"
        request_payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt_text}],
                }
            ],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_output_tokens,
            },
        }
        body = json.dumps(request_payload).encode("utf-8")
        request = urllib_request.Request(
            url,
            data=body,
            headers={
                "content-type": "application/json",
                "x-goog-api-key": self.api_key,
            },
            method="POST",
        )

        with urllib_request.urlopen(request) as response:
            response_payload = json.loads(response.read().decode("utf-8"))

        candidates = response_payload.get("candidates", []) or []
        candidate = candidates[0] if candidates else {}
        content = candidate.get("content", {}).get("parts", [])
        text = _extract_message_text(content)
        usage = response_payload.get("usageMetadata", {}) or {}

        return GenerationResponse(
            text=text,
            payload={
                "provider_type": "gemini-native",
                "task_id": task_id,
                "trial_index": trial_index,
                "api_base_url": base_url,
                "model_name": self.model_name,
                "temperature": self.temperature,
                "max_output_tokens": self.max_output_tokens,
                "response_id": response_payload.get("responseId"),
                "response_model": response_payload.get("modelVersion"),
                "finish_reason": candidate.get("finishReason"),
                "prompt_tokens": usage.get("promptTokenCount"),
                "completion_tokens": usage.get("candidatesTokenCount"),
                "total_tokens": usage.get("totalTokenCount"),
            },
        )


@dataclass(frozen=True)
class AnthropicMessagesClient:
    """Model client backed by Anthropic's native Messages API."""

    model_name: str
    api_key: str
    api_base_url: str | None = None
    anthropic_version: str = "2023-06-01"
    temperature: float = 0.0
    max_output_tokens: int = 2048

    @property
    def model_id(self) -> str:
        return self.model_name

    def generate(self, *, prompt_text: str, task_id: str, trial_index: int) -> GenerationResponse:
        base_url = (self.api_base_url or "https://api.anthropic.com").rstrip("/")
        url = f"{base_url}/v1/messages"
        request_payload = {
            "model": self.model_name,
            "max_tokens": self.max_output_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt_text}],
        }
        body = json.dumps(request_payload).encode("utf-8")
        request = urllib_request.Request(
            url,
            data=body,
            headers={
                "content-type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": self.anthropic_version,
            },
            method="POST",
        )

        with urllib_request.urlopen(request) as response:
            response_payload = json.loads(response.read().decode("utf-8"))

        content = response_payload.get("content", "")
        text = _extract_message_text(content)
        usage = response_payload.get("usage", {}) or {}
        prompt_tokens = usage.get("input_tokens")
        completion_tokens = usage.get("output_tokens")
        total_tokens = None
        if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
            total_tokens = prompt_tokens + completion_tokens

        return GenerationResponse(
            text=text,
            payload={
                "provider_type": "anthropic-native",
                "task_id": task_id,
                "trial_index": trial_index,
                "api_base_url": base_url,
                "anthropic_version": self.anthropic_version,
                "model_name": self.model_name,
                "temperature": self.temperature,
                "max_output_tokens": self.max_output_tokens,
                "response_id": response_payload.get("id"),
                "response_model": response_payload.get("model"),
                "finish_reason": response_payload.get("stop_reason"),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        )
