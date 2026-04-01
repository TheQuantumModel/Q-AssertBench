import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import types
import unittest
from unittest import mock

from qasserbench.generation.clients import (
    AnthropicMessagesClient,
    GeminiGenerativeLanguageClient,
    OpenAICompatibleClient,
)
from scripts.run_generation import build_generation_client, load_local_env_file


class _FakeUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str, finish_reason: str = "stop") -> None:
        self.message = _FakeMessage(content)
        self.finish_reason = finish_reason


class _FakeCompletions:
    def __init__(self, recorder: dict[str, object]) -> None:
        self._recorder = recorder

    def create(self, **kwargs):
        self._recorder.update(kwargs)
        return types.SimpleNamespace(
            id="resp_123",
            model=kwargs["model"],
            choices=[_FakeChoice("```python\nassert True\n```")],
            usage=_FakeUsage(prompt_tokens=12, completion_tokens=5),
        )


class _FakeOpenAI:
    def __init__(self, recorder: dict[str, object]) -> None:
        self.chat = types.SimpleNamespace(completions=_FakeCompletions(recorder))


class _FakeAnthropicHTTPResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class OpenAICompatibleClientTests(unittest.TestCase):
    def test_load_local_env_file_sets_missing_values_without_overwriting_existing_env(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / ".env.local"
            env_path.write_text(
                "OPENROUTER_API_KEY=file-secret\n"
                "QAB_API_BASE_URL=https://example.invalid/v1\n",
                encoding="utf-8",
            )

            with mock.patch.dict(
                "os.environ",
                {"OPENROUTER_API_KEY": "existing-secret"},
                clear=True,
            ):
                loaded_path = load_local_env_file(env_path)

                self.assertEqual(loaded_path, env_path.resolve())
                self.assertEqual("existing-secret", os.environ["OPENROUTER_API_KEY"])
                self.assertEqual("https://example.invalid/v1", os.environ["QAB_API_BASE_URL"])

    def test_build_generation_client_reads_openai_compatible_settings_from_env(self) -> None:
        args = argparse.Namespace(
            client="openai-compatible",
            model=None,
            model_id="unused-static-model",
            response_text="unused",
            api_base_url=None,
            api_key_env="QAB_API_KEY",
            temperature=0.3,
            max_output_tokens=400,
        )

        with mock.patch.dict(
            "os.environ",
            {
                "QAB_MODEL": "env-model",
                "QAB_API_KEY": "env-secret",
                "QAB_API_BASE_URL": "https://example.invalid/v1",
            },
            clear=False,
        ):
            with mock.patch("scripts.run_generation.OpenAICompatibleClient") as client_cls:
                build_generation_client(args)

        client_cls.assert_called_once_with(
            model_name="env-model",
            api_key="env-secret",
            api_base_url="https://example.invalid/v1",
            temperature=0.3,
            max_output_tokens=400,
        )

    def test_build_generation_client_reads_anthropic_native_settings_from_env(self) -> None:
        args = argparse.Namespace(
            client="anthropic-native",
            model=None,
            model_id="unused-static-model",
            response_text="unused",
            api_base_url=None,
            api_key_env="Claude_API_KEY",
            anthropic_version="2023-06-01",
            temperature=0.7,
            max_output_tokens=500,
        )

        with mock.patch.dict(
            "os.environ",
            {
                "QAB_MODEL": "claude-sonnet-4-20250514",
                "Claude_API_KEY": "claude-secret",
                "QAB_API_BASE_URL": "https://api.anthropic.com",
            },
            clear=False,
        ):
            with mock.patch("scripts.run_generation.AnthropicMessagesClient") as client_cls:
                build_generation_client(args)

        client_cls.assert_called_once_with(
            model_name="claude-sonnet-4-20250514",
            api_key="claude-secret",
            api_base_url="https://api.anthropic.com",
            anthropic_version="2023-06-01",
            temperature=0.7,
            max_output_tokens=500,
        )

    def test_build_generation_client_reads_gemini_native_settings_from_env(self) -> None:
        args = argparse.Namespace(
            client="gemini-native",
            model=None,
            model_id="unused-static-model",
            response_text="unused",
            api_base_url=None,
            api_key_env="Gemini_API_KEY",
            anthropic_version="unused",
            temperature=0.8,
            max_output_tokens=600,
        )

        with mock.patch.dict(
            "os.environ",
            {
                "QAB_MODEL": "gemini-3.1-flash-lite-preview",
                "Gemini_API_KEY": "gemini-secret",
                "GEMINI_API_BASE_URL": "https://generativelanguage.googleapis.com",
            },
            clear=False,
        ):
            with mock.patch("scripts.run_generation.GeminiGenerativeLanguageClient") as client_cls:
                build_generation_client(args)

        client_cls.assert_called_once_with(
            model_name="gemini-3.1-flash-lite-preview",
            api_key="gemini-secret",
            api_base_url="https://generativelanguage.googleapis.com",
            temperature=0.8,
            max_output_tokens=600,
        )

    def test_calls_chat_completions_with_rendered_prompt(self) -> None:
        init_calls: list[dict[str, object]] = []
        request_calls: dict[str, object] = {}

        def fake_openai_factory(*, api_key: str, base_url: str | None = None):
            init_calls.append({"api_key": api_key, "base_url": base_url})
            return _FakeOpenAI(request_calls)

        fake_module = types.SimpleNamespace(OpenAI=fake_openai_factory)

        with mock.patch(
            "qasserbench.generation.clients._load_openai_sdk",
            return_value=fake_module,
        ):
            client = OpenAICompatibleClient(
                model_name="test-model",
                api_key="secret",
                api_base_url="https://example.invalid/v1",
                temperature=0.2,
                max_output_tokens=256,
            )
            response = client.generate(
                prompt_text="rendered prompt text",
                task_id="QAB01",
                trial_index=2,
            )

        self.assertEqual(init_calls, [{"api_key": "secret", "base_url": "https://example.invalid/v1"}])
        self.assertEqual(request_calls["model"], "test-model")
        self.assertEqual(request_calls["messages"], [{"role": "user", "content": "rendered prompt text"}])
        self.assertEqual(request_calls["temperature"], 0.2)
        self.assertEqual(request_calls["max_tokens"], 256)
        self.assertEqual(response.text, "```python\nassert True\n```")
        self.assertEqual(response.payload["finish_reason"], "stop")
        self.assertEqual(response.payload["provider_type"], "openai-compatible")
        self.assertEqual(response.payload["task_id"], "QAB01")
        self.assertEqual(response.payload["trial_index"], 2)
        self.assertEqual(response.payload["temperature"], 0.2)
        self.assertEqual(response.payload["max_output_tokens"], 256)
        self.assertEqual(response.payload["prompt_tokens"], 12)
        self.assertEqual(response.payload["completion_tokens"], 5)
        self.assertEqual(response.payload["total_tokens"], 17)

    def test_uses_max_completion_tokens_for_official_openai_gpt5_models(self) -> None:
        init_calls: list[dict[str, object]] = []
        request_calls: dict[str, object] = {}

        def fake_openai_factory(*, api_key: str, base_url: str | None = None):
            init_calls.append({"api_key": api_key, "base_url": base_url})
            return _FakeOpenAI(request_calls)

        fake_module = types.SimpleNamespace(OpenAI=fake_openai_factory)

        with mock.patch(
            "qasserbench.generation.clients._load_openai_sdk",
            return_value=fake_module,
        ):
            client = OpenAICompatibleClient(
                model_name="gpt-5.4",
                api_key="secret",
                api_base_url="https://api.openai.com/v1",
                temperature=1.0,
                max_output_tokens=2048,
            )
            client.generate(
                prompt_text="rendered prompt text",
                task_id="QAB19",
                trial_index=1,
            )

        self.assertEqual(request_calls["model"], "gpt-5.4")
        self.assertEqual(request_calls["temperature"], 1.0)
        self.assertEqual(request_calls["max_completion_tokens"], 2048)
        self.assertNotIn("max_tokens", request_calls)

    def test_raises_helpful_error_when_openai_sdk_is_missing(self) -> None:
        with mock.patch(
            "qasserbench.generation.clients._load_openai_sdk",
            side_effect=ModuleNotFoundError("No module named 'openai'"),
        ):
            with self.assertRaisesRegex(RuntimeError, "openai package is required"):
                OpenAICompatibleClient(
                    model_name="test-model",
                    api_key="secret",
                )


class AnthropicMessagesClientTests(unittest.TestCase):
    def test_posts_messages_request_and_extracts_usage(self) -> None:
        captured: dict[str, object] = {}

        def fake_urlopen(request):
            captured["url"] = request.full_url
            captured["method"] = request.get_method()
            captured["headers"] = dict(request.header_items())
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return _FakeAnthropicHTTPResponse(
                {
                    "id": "msg_123",
                    "model": "claude-sonnet-4-20250514",
                    "content": [{"type": "text", "text": "```python\nassert True\n```"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 21, "output_tokens": 8},
                }
            )

        with mock.patch("qasserbench.generation.clients.urllib_request.urlopen", side_effect=fake_urlopen):
            client = AnthropicMessagesClient(
                model_name="claude-sonnet-4-20250514",
                api_key="claude-secret",
                api_base_url="https://api.anthropic.com",
                anthropic_version="2023-06-01",
                temperature=1.0,
                max_output_tokens=2048,
            )
            response = client.generate(
                prompt_text="rendered prompt text",
                task_id="QAB19",
                trial_index=1,
            )

        self.assertEqual(captured["url"], "https://api.anthropic.com/v1/messages")
        self.assertEqual(captured["method"], "POST")
        self.assertEqual(captured["body"]["model"], "claude-sonnet-4-20250514")
        self.assertEqual(captured["body"]["messages"], [{"role": "user", "content": "rendered prompt text"}])
        self.assertEqual(captured["body"]["temperature"], 1.0)
        self.assertEqual(captured["body"]["max_tokens"], 2048)
        headers = {str(k).lower(): v for k, v in captured["headers"].items()}
        self.assertEqual(headers["x-api-key"], "claude-secret")
        self.assertEqual(headers["anthropic-version"], "2023-06-01")
        self.assertEqual(response.text, "```python\nassert True\n```")
        self.assertEqual(response.payload["provider_type"], "anthropic-native")
        self.assertEqual(response.payload["finish_reason"], "end_turn")
        self.assertEqual(response.payload["prompt_tokens"], 21)
        self.assertEqual(response.payload["completion_tokens"], 8)
        self.assertEqual(response.payload["total_tokens"], 29)


class GeminiGenerativeLanguageClientTests(unittest.TestCase):
    def test_posts_generate_content_request_and_extracts_usage(self) -> None:
        captured: dict[str, object] = {}

        def fake_urlopen(request):
            captured["url"] = request.full_url
            captured["method"] = request.get_method()
            captured["headers"] = dict(request.header_items())
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return _FakeAnthropicHTTPResponse(
                {
                    "responseId": "gem_123",
                    "modelVersion": "gemini-3.1-flash-lite-preview",
                    "candidates": [
                        {
                            "content": {
                                "parts": [{"text": "```python\nassert True\n```"}],
                            },
                            "finishReason": "STOP",
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 18,
                        "candidatesTokenCount": 7,
                        "totalTokenCount": 25,
                    },
                }
            )

        with mock.patch("qasserbench.generation.clients.urllib_request.urlopen", side_effect=fake_urlopen):
            client = GeminiGenerativeLanguageClient(
                model_name="gemini-3.1-flash-lite-preview",
                api_key="gemini-secret",
                api_base_url="https://generativelanguage.googleapis.com",
                temperature=1.0,
                max_output_tokens=2048,
            )
            response = client.generate(
                prompt_text="rendered prompt text",
                task_id="QAB19",
                trial_index=1,
            )

        self.assertEqual(
            captured["url"],
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite-preview:generateContent",
        )
        self.assertEqual(captured["method"], "POST")
        self.assertEqual(
            captured["body"]["contents"],
            [{"role": "user", "parts": [{"text": "rendered prompt text"}]}],
        )
        self.assertEqual(
            captured["body"]["generationConfig"],
            {"temperature": 1.0, "maxOutputTokens": 2048},
        )
        headers = {str(k).lower(): v for k, v in captured["headers"].items()}
        self.assertEqual(headers["x-goog-api-key"], "gemini-secret")
        self.assertEqual(response.text, "```python\nassert True\n```")
        self.assertEqual(response.payload["provider_type"], "gemini-native")
        self.assertEqual(response.payload["finish_reason"], "STOP")
        self.assertEqual(response.payload["prompt_tokens"], 18)
        self.assertEqual(response.payload["completion_tokens"], 7)
        self.assertEqual(response.payload["total_tokens"], 25)


if __name__ == "__main__":
    unittest.main()
