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


def _renumber_trial_indices(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    renumbered: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        updated = dict(record)
        updated["trial_index"] = index
        raw_payload = dict(updated.get("raw_payload", {}))
        raw_payload["trial_index"] = index
        updated["raw_payload"] = raw_payload
        renumbered.append(updated)
    return renumbered


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
    api_key_env: str,
    target_trials: int,
    temperature: float,
    max_output_tokens: int,
    concurrency: int,
    supplement_tag: str,
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    model_dir = Path(model_dir).resolve()
    tasks_dir = model_dir / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    supplement_dir = model_dir / "supplements" / supplement_tag
    supplement_dir.mkdir(parents=True, exist_ok=True)

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

        supplement_output = supplement_dir / f"{task_id}.jsonl"
        print(
            f"[{model_name}] task {task_index}/{len(task_ids)} {task_id} BACKFILL deficit={deficit}",
            flush=True,
        )
        result = subprocess.run(
            [
                python,
                generation_script,
                str(supplement_output),
                "--client",
                "openai-compatible",
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
                "--concurrency",
                str(concurrency),
                "--temperature",
                str(temperature),
                "--max-output-tokens",
                str(max_output_tokens),
            ],
            cwd=project_root,
            text=True,
            capture_output=True,
        )
        if result.stdout.strip():
            print(result.stdout.strip(), flush=True)
        if result.returncode != 0:
            if result.stderr.strip():
                print(result.stderr.strip(), flush=True)
            raise SystemExit(f"FAILED model={model_name} task={task_id} code={result.returncode}")

        merged_records = _read_task_records(tasks_dir, task_id) + read_generation_records(supplement_output)
        merged_records = _renumber_trial_indices(merged_records)
        write_generation_records_jsonl(merged_records, tasks_dir / f"{task_id}.jsonl")
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
        "--api-key-env",
        default="OPENROUTER_API_KEY",
        help="Environment variable containing the API key",
    )
    parser.add_argument("--target-trials", type=int, default=20, help="Desired trials per task")
    parser.add_argument("--temperature", type=float, default=1.0, help="Generation temperature")
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=2048,
        help="Maximum completion tokens for backfill generation",
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
        api_key_env=args.api_key_env,
        target_trials=args.target_trials,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        concurrency=args.concurrency,
        supplement_tag=args.supplement_tag,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
