"""Repeated generation driver for benchmark trials."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from qasserbench.benchmark.loader import load_task_assets, load_task_manifest
from qasserbench.generation.clients import ModelClient
from qasserbench.generation.prompting import PROMPT_TEMPLATE_VERSION, render_task_prompt


def discover_task_manifests(
    tasks_root: str | Path,
    task_ids: Sequence[str] | None = None,
) -> list[Path]:
    """Discover task manifests, optionally filtered by task id."""

    root = Path(tasks_root).resolve()
    selected = {task_id.upper() for task_id in task_ids} if task_ids else None

    manifest_paths: list[Path] = []
    for manifest_path in sorted(root.glob("*/task.yaml")):
        task = load_task_manifest(manifest_path)
        if selected and task.task_id.upper() not in selected:
            continue
        manifest_paths.append(manifest_path.resolve())

    return manifest_paths


def write_generation_records(records: Iterable[dict[str, Any]], output_path: str | Path) -> Path:
    """Persist raw generation records as JSONL."""

    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    return destination


def _append_generation_record(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, sort_keys=True) + "\n")
    handle.flush()


def read_generation_records(input_path: str | Path) -> list[dict[str, Any]]:
    """Load raw generation records from a JSONL file."""

    source = Path(input_path).resolve()
    records: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = line.strip()
            if not payload:
                continue
            records.append(json.loads(payload))
    return records


def _standardized_generation_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "generation_temperature": payload.get("temperature"),
        "generation_max_output_tokens": payload.get("max_output_tokens"),
        "prompt_tokens": payload.get("prompt_tokens"),
        "completion_tokens": payload.get("completion_tokens"),
        "total_tokens": payload.get("total_tokens"),
    }


def run_generation_trials(
    *,
    manifest_paths: Sequence[str | Path],
    client: ModelClient,
    trial_count: int,
    output_path: str | Path,
    record_model_id: str | None = None,
    max_concurrency: int = 1,
) -> Path:
    """Run repeated generation trials and persist the raw responses."""

    if trial_count < 1:
        raise ValueError("trial_count must be at least 1.")
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be at least 1.")

    jobs: list[dict[str, Any]] = []
    for manifest_path in manifest_paths:
        assets = load_task_assets(manifest_path)
        task = assets.task
        rendered_prompt = render_task_prompt(assets)

        for trial_index in range(1, trial_count + 1):
            jobs.append(
                {
                    "manifest_path": str(Path(manifest_path).resolve()),
                    "task_id": task.task_id,
                    "task_family": task.family,
                    "property_type": task.property_type,
                    "trial_index": trial_index,
                    "rendered_prompt": rendered_prompt,
                }
            )

    def build_record(job: dict[str, Any]) -> dict[str, Any]:
        response = client.generate(
            prompt_text=str(job["rendered_prompt"]),
            task_id=str(job["task_id"]),
            trial_index=int(job["trial_index"]),
        )
        standardized_metadata = _standardized_generation_metadata(dict(response.payload))
        return {
            "model_id": record_model_id or client.model_id,
            "provider_model_id": client.model_id,
            "task_id": str(job["task_id"]),
            "trial_index": int(job["trial_index"]),
            "task_family": str(job["task_family"]),
            "property_type": str(job["property_type"]),
            "manifest_path": str(job["manifest_path"]),
            "prompt_version": PROMPT_TEMPLATE_VERSION,
            "raw_response": response.text,
            "raw_payload": dict(response.payload),
            **standardized_metadata,
        }

    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open("w", encoding="utf-8") as handle:
        if max_concurrency == 1 or len(jobs) <= 1:
            for job in jobs:
                _append_generation_record(handle, build_record(job))
        else:
            with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
                futures = [executor.submit(build_record, job) for job in jobs]
                for future in as_completed(futures):
                    _append_generation_record(handle, future.result())

    return destination
