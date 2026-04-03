"""Backfill curated generation datasets up to a target trial count per task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any

import yaml

from qasserbench.generation.curation import (
    missing_trial_counts_by_task,
    write_generation_records_jsonl,
)
from qasserbench.generation.driver import read_generation_records


def _task_ids(tasks_root: str | Path) -> list[str]:
    tasks_root = Path(tasks_root).resolve()
    task_ids: list[str] = []
    for manifest_path in sorted(tasks_root.glob("*/task.yaml")):
        payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        task_ids.append(str(payload["task_id"]))
    return task_ids


def _read_task_records(tasks_dir: Path, task_id: str) -> list[dict[str, Any]]:
    task_path = tasks_dir / f"{task_id}.jsonl"
    if not task_path.exists():
        return []
    return read_generation_records(task_path)


def _merge_records_by_trial(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[int, dict[str, Any]] = {}
    for record in records:
        merged[int(record["trial_index"])] = record
    return [merged[index] for index in sorted(merged)]

def _next_trial_index(records: list[dict[str, Any]]) -> int:
    if not records:
        return 1
    return max(int(record["trial_index"]) for record in records) + 1


def _rewrite_combined_file(model_dir: Path, task_ids: list[str]) -> int:
    tasks_dir = model_dir / "tasks"
    combined_records: list[dict[str, Any]] = []
    for task_id in task_ids:
        combined_records.extend(_read_task_records(tasks_dir, task_id))
    write_generation_records_jsonl(combined_records, model_dir / "generation_records.jsonl")
    return len(combined_records)


def backfill_model_directory(
    *,
    model_dir: str | Path,
    model_name: str,
    tasks_root: str | Path,
    client_type: str,
    api_key_env: str,
    api_base_url: str | None,
    anthropic_version: str,
    target_trials: int,
    temperature: float,
    max_output_tokens: int,
    request_timeout_seconds: float,
    concurrency: int,
    supplement_tag: str,
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    model_dir = Path(model_dir).resolve()
    tasks_dir = model_dir / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)

    task_ids = _task_ids(tasks_root)
    existing_records: list[dict[str, Any]] = []
    for task_id in task_ids:
        existing_records.extend(_read_task_records(tasks_dir, task_id))

    deficits = missing_trial_counts_by_task(
        existing_records,
        target_trials=target_trials,
        task_ids=task_ids,
    )

    python = str(project_root / ".venv" / "bin" / "python")
    generation_script = str(project_root / "scripts" / "run_generation.py")

    for task_index, task_id in enumerate(task_ids, start=1):
        deficit = deficits.get(task_id, 0)
        if deficit < 1:
            print(
                f"[{model_name}] task {task_index}/{len(task_ids)} {task_id} SKIP deficit=0",
                flush=True,
            )
            continue

        task_output = tasks_dir / f"{task_id}.jsonl"
        existing_task_records = _merge_records_by_trial(_read_task_records(tasks_dir, task_id))
        next_trial_index = _next_trial_index(existing_task_records)
        print(
            f"[{model_name}] task {task_index}/{len(task_ids)} {task_id} BACKFILL deficit={deficit}",
            flush=True,
        )
        command = [
            python,
            generation_script,
            str(task_output),
            "--client",
            client_type,
            "--model",
            model_name,
            "--model-id",
            model_name,
            "--tasks-root",
            str(Path(tasks_root).resolve()),
            "--api-key-env",
            api_key_env,
            "--task-id",
            task_id,
            "--trials",
            str(deficit),
            "--trial-start-index",
            str(next_trial_index),
            "--concurrency",
            str(concurrency),
            "--append",
            "--temperature",
            str(temperature),
            "--max-output-tokens",
            str(max_output_tokens),
            "--request-timeout-seconds",
            str(request_timeout_seconds),
            "--anthropic-version",
            anthropic_version,
        ]
        if api_base_url:
            command.extend(["--api-base-url", api_base_url])
        result = subprocess.run(
            command,
            cwd=project_root,
            text=True,
            capture_output=True,
        )
        if result.stdout.strip():
            print(result.stdout.strip(), flush=True)
        if result.returncode != 0:
            normalized_records = _merge_records_by_trial(_read_task_records(tasks_dir, task_id))
            if normalized_records:
                write_generation_records_jsonl(normalized_records, task_output)
            _rewrite_combined_file(model_dir, task_ids)
            if result.stderr.strip():
                print(result.stderr.strip(), flush=True)
            raise SystemExit(f"FAILED model={model_name} task={task_id} code={result.returncode}")

        merged_records = _merge_records_by_trial(_read_task_records(tasks_dir, task_id))
        write_generation_records_jsonl(merged_records, task_output)
        total_records = _rewrite_combined_file(model_dir, task_ids)
        print(
            f"[{model_name}] task {task_index}/{len(task_ids)} {task_id} DONE task_records={len(merged_records)} accumulated_records={total_records}",
            flush=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill curated generation datasets.")
    parser.add_argument("--model-dir", required=True, type=Path, help="Curated model directory")
    parser.add_argument("--model-name", required=True, help="Provider model name")
    parser.add_argument(
        "--tasks-root",
        type=Path,
        default=Path("benchmark_data/tasks"),
        help="Root directory containing benchmark tasks",
    )
    parser.add_argument(
        "--client",
        choices=("openai-compatible", "anthropic-native", "gemini-native"),
        default="openai-compatible",
        help="Generation client backend used for supplement runs",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENROUTER_API_KEY",
        help="Environment variable containing the API key",
    )
    parser.add_argument(
        "--api-base-url",
        default=None,
        help="Optional base URL for the selected client",
    )
    parser.add_argument(
        "--anthropic-version",
        default="2023-06-01",
        help="Anthropic API version when using --client anthropic-native",
    )
    parser.add_argument("--target-trials", type=int, default=20, help="Desired trials per task")
    parser.add_argument("--temperature", type=float, default=1.0, help="Generation temperature")
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=2048,
        help="Maximum completion tokens for backfill generation",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=120.0,
        help="HTTP request timeout used by the generation client",
    )
    parser.add_argument("--concurrency", type=int, default=5, help="Concurrent generation requests")
    parser.add_argument(
        "--supplement-tag",
        default="max_tokens_2048",
        help="Directory label used to store supplemental raw generations",
    )
    args = parser.parse_args()

    backfill_model_directory(
        model_dir=args.model_dir,
        model_name=args.model_name,
        tasks_root=args.tasks_root,
        client_type=args.client,
        api_key_env=args.api_key_env,
        api_base_url=args.api_base_url,
        anthropic_version=args.anthropic_version,
        target_trials=args.target_trials,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        request_timeout_seconds=args.request_timeout_seconds,
        concurrency=args.concurrency,
        supplement_tag=args.supplement_tag,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
