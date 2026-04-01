"""Run repeated raw-generation trials for selected benchmark tasks."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
from typing import Any, Sequence

import yaml

from qasserbench.benchmark.loader import load_task_manifest
from qasserbench.generation.clients import (
    AnthropicMessagesClient,
    GeminiGenerativeLanguageClient,
    ModelClient,
    OpenAICompatibleClient,
    StaticResponseClient,
)
from qasserbench.generation.driver import discover_task_manifests, run_generation_trials


def load_local_env_file(env_path: str | Path, *, override: bool = False) -> Path | None:
    source = Path(env_path).resolve()
    if not source.exists():
        return None

    for raw_line in source.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if override or key not in os.environ:
            os.environ[key] = value

    return source


def load_default_local_env(*, override: bool = False) -> Path | None:
    project_root = Path(__file__).resolve().parents[1]
    return load_local_env_file(project_root / ".env.local", override=override)


def run_generation_experiment(
    *,
    tasks_root: str | Path,
    output_path: str | Path,
    client: ModelClient,
    task_ids: Sequence[str] | None = None,
    trial_count: int = 1,
    record_model_id: str | None = None,
    max_concurrency: int = 1,
) -> Path:
    """Run repeated generation for selected tasks and persist raw outputs."""

    manifest_paths = discover_task_manifests(tasks_root, task_ids=task_ids)
    return run_generation_trials(
        manifest_paths=manifest_paths,
        client=client,
        trial_count=trial_count,
        output_path=output_path,
        record_model_id=record_model_id,
        max_concurrency=max_concurrency,
    )


def load_generation_manifest(manifest_path: str | Path) -> dict[str, Any]:
    source = Path(manifest_path).resolve()
    with source.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("Generation manifest must be a YAML mapping.")
    return payload


def _sanitize_model_id(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id).strip("_")


def resolve_manifest_task_ids(manifest: dict[str, Any]) -> list[str]:
    defaults = manifest.get("defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError("Manifest defaults must be a mapping.")
    tasks_root = defaults.get("tasks_root", "benchmark_data/tasks")

    task_selection = manifest.get("task_selection", {}) or {}
    if not isinstance(task_selection, dict):
        raise ValueError("Manifest task_selection must be a mapping.")
    mode = str(task_selection.get("mode", "all")).lower()
    selected = [str(task_id).upper() for task_id in task_selection.get("task_ids", [])]

    all_manifest_paths = discover_task_manifests(tasks_root)
    all_task_ids = [load_task_manifest(path).task_id for path in all_manifest_paths]
    selected_set = set(selected)

    if mode == "all":
        return all_task_ids
    if mode == "include":
        return [task_id for task_id in all_task_ids if task_id.upper() in selected_set]
    if mode == "exclude":
        return [task_id for task_id in all_task_ids if task_id.upper() not in selected_set]

    raise ValueError(f"Unsupported task selection mode: {mode}")


def _build_generation_client_from_manifest(
    *,
    client_type: str,
    api_config: dict[str, Any],
    defaults: dict[str, Any],
    model_config: dict[str, Any],
) -> ModelClient:
    model_name = str(model_config["model_id"])
    if client_type == "static":
        response_text = str(
            model_config.get(
                "response_text",
                defaults.get("response_text", "```python\nassert True\n```"),
            )
        )
        return StaticResponseClient(
            model_name=model_name,
            response_text=response_text,
        )

    if client_type != "openai-compatible":
        if client_type == "gemini-native":
            api_key_env = str(api_config.get("api_key_env", "Gemini_API_KEY"))
            api_key = os.environ.get(api_key_env)
            if not api_key:
                raise ValueError(
                    f"Gemini-native manifest requires environment variable {api_key_env}."
                )

            return GeminiGenerativeLanguageClient(
                model_name=model_name,
                api_key=api_key,
                api_base_url=api_config.get("base_url") or os.environ.get("GEMINI_API_BASE_URL"),
                temperature=float(
                    model_config.get("temperature", defaults.get("temperature", 0.0))
                ),
                max_output_tokens=int(
                    model_config.get("max_output_tokens", defaults.get("max_output_tokens", 2048))
                ),
            )

        if client_type != "anthropic-native":
            raise ValueError(f"Unsupported manifest client type: {client_type}")

        api_key_env = str(api_config.get("api_key_env", "Claude_API_KEY"))
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(
                f"Anthropic-native manifest requires environment variable {api_key_env}."
            )

        return AnthropicMessagesClient(
            model_name=model_name,
            api_key=api_key,
            api_base_url=api_config.get("base_url"),
            anthropic_version=str(
                model_config.get(
                    "anthropic_version",
                    defaults.get("anthropic_version", "2023-06-01"),
                )
            ),
            temperature=float(model_config.get("temperature", defaults.get("temperature", 0.0))),
            max_output_tokens=int(
                model_config.get("max_output_tokens", defaults.get("max_output_tokens", 2048))
            ),
        )

    api_key_env = str(api_config.get("api_key_env", "QAB_API_KEY"))
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise ValueError(
            f"OpenAI-compatible manifest requires environment variable {api_key_env}."
        )

    return OpenAICompatibleClient(
        model_name=model_name,
        api_key=api_key,
        api_base_url=api_config.get("base_url"),
        temperature=float(model_config.get("temperature", defaults.get("temperature", 0.0))),
        max_output_tokens=int(
            model_config.get("max_output_tokens", defaults.get("max_output_tokens", 2048))
        ),
    )


def run_manifest_generation_experiment(manifest_path: str | Path) -> list[Path]:
    manifest = load_generation_manifest(manifest_path)
    defaults = manifest.get("defaults", {}) or {}
    api_config = manifest.get("api", {}) or {}
    models = manifest.get("models", []) or []
    client_type = str(manifest.get("client", "openai-compatible"))

    if not isinstance(defaults, dict):
        raise ValueError("Manifest defaults must be a mapping.")
    if not isinstance(api_config, dict):
        raise ValueError("Manifest api must be a mapping.")
    if not isinstance(models, list):
        raise ValueError("Manifest models must be a list.")

    task_ids = resolve_manifest_task_ids(manifest)
    tasks_root = Path(str(defaults.get("tasks_root", "benchmark_data/tasks"))).resolve()
    output_dir = Path(str(defaults.get("output_dir"))).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_manifest = {
        **manifest,
        "defaults": {
            **defaults,
            "tasks_root": str(tasks_root),
            "output_dir": str(output_dir),
        },
        "resolved_task_ids": list(task_ids),
    }
    with (output_dir / "resolved_manifest.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(resolved_manifest, handle, sort_keys=False)

    written_paths: list[Path] = []
    for model_entry in models:
        if not isinstance(model_entry, dict):
            raise ValueError("Each model entry must be a mapping.")
        if model_entry.get("enabled", True) is False:
            continue

        client = _build_generation_client_from_manifest(
            client_type=client_type,
            api_config=api_config,
            defaults=defaults,
            model_config=model_entry,
        )
        record_model_id = str(model_entry.get("run_id", model_entry["model_id"]))
        model_output_dir = output_dir / _sanitize_model_id(record_model_id)
        model_output_dir.mkdir(parents=True, exist_ok=True)
        output_path = model_output_dir / "generation_records.jsonl"
        trial_count = int(model_entry.get("trials", defaults.get("trials", 1)))
        max_concurrency = int(model_entry.get("concurrency", defaults.get("concurrency", 1)))
        written_paths.append(
            run_generation_experiment(
                tasks_root=tasks_root,
                output_path=output_path,
                client=client,
                task_ids=task_ids,
                trial_count=trial_count,
                record_model_id=record_model_id,
                max_concurrency=max_concurrency,
            )
        )

    return written_paths


def build_generation_client(args: argparse.Namespace) -> ModelClient:
    client_type = str(args.client)
    if client_type == "static":
        return StaticResponseClient(
            model_name=args.model_id,
            response_text=args.response_text,
        )

    model_name = args.model or os.environ.get("QAB_MODEL")
    if not model_name:
        raise ValueError(f"{client_type} client requires --model or QAB_MODEL.")

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise ValueError(
            f"{client_type} client requires environment variable {args.api_key_env}."
        )

    if client_type == "gemini-native":
        api_base_url = (
            args.api_base_url
            or os.environ.get("GEMINI_API_BASE_URL")
            or "https://generativelanguage.googleapis.com"
        )
        return GeminiGenerativeLanguageClient(
            model_name=model_name,
            api_key=api_key,
            api_base_url=api_base_url,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
        )

    api_base_url = args.api_base_url or os.environ.get("QAB_API_BASE_URL")
    if client_type == "anthropic-native":
        return AnthropicMessagesClient(
            model_name=model_name,
            api_key=api_key,
            api_base_url=api_base_url or "https://api.anthropic.com",
            anthropic_version=args.anthropic_version,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
        )

    if client_type != "openai-compatible":
        raise ValueError(f"Unsupported client type: {client_type}")

    return OpenAICompatibleClient(
        model_name=model_name,
        api_key=api_key,
        api_base_url=api_base_url,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
    )


def main() -> int:
    load_default_local_env()
    parser = argparse.ArgumentParser(description="Run repeated raw-generation trials.")
    parser.add_argument("output", nargs="?", type=Path, help="Path to generation_records.jsonl")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="YAML experiment manifest for multi-model generation runs",
    )
    parser.add_argument(
        "--client",
        choices=("static", "openai-compatible", "anthropic-native", "gemini-native"),
        default="static",
        help="Generation client backend",
    )
    parser.add_argument(
        "--tasks-root",
        type=Path,
        default=Path("benchmark_data/tasks"),
        help="Root directory containing task folders with task.yaml files",
    )
    parser.add_argument(
        "--task-id",
        dest="task_ids",
        action="append",
        default=None,
        help="Specific task id to include; may be repeated",
    )
    parser.add_argument("--trials", type=int, default=1, help="Number of repeated trials per task")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Maximum number of concurrent generation requests",
    )
    parser.add_argument("--model-id", default="static-model", help="Model identifier to record")
    parser.add_argument("--model", default=None, help="Model name for OpenAI-compatible client")
    parser.add_argument(
        "--api-base-url",
        default=None,
        help="Optional base URL for OpenAI-compatible client; defaults to QAB_API_BASE_URL",
    )
    parser.add_argument(
        "--api-key-env",
        default="QAB_API_KEY",
        help="Environment variable containing the API key for the selected generation client",
    )
    parser.add_argument(
        "--anthropic-version",
        default="2023-06-01",
        help="Anthropic API version header when using --client anthropic-native",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for OpenAI-compatible generation",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=2048,
        help="Maximum completion tokens for OpenAI-compatible generation",
    )
    parser.add_argument(
        "--response-text",
        default="```python\nassert True\n```",
        help="Static response text emitted for every generation trial",
    )
    args = parser.parse_args()

    if args.manifest is not None:
        output_paths = run_manifest_generation_experiment(args.manifest)
        print(f"wrote generation records to {len(output_paths)} model output(s)")
        return 0

    if args.output is None:
        raise ValueError("Direct generation mode requires the output path argument.")

    client = build_generation_client(args)
    output_path = run_generation_experiment(
        tasks_root=args.tasks_root,
        output_path=args.output,
        client=client,
        task_ids=args.task_ids,
        trial_count=args.trials,
        max_concurrency=args.concurrency,
    )
    print(f"wrote generation records to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
