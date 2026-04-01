"""Helpers for curating stored generation records."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


def generation_finish_reason(record: dict[str, Any]) -> str | None:
    payload = record.get("raw_payload") or {}
    if not isinstance(payload, dict):
        return None
    finish_reason = payload.get("finish_reason")
    return finish_reason if isinstance(finish_reason, str) else None


def curate_generation_records(
    records: Sequence[dict[str, Any]],
    *,
    drop_missing_finish_reason: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split generation records into kept and dropped subsets."""

    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []

    for record in records:
        finish_reason = generation_finish_reason(record)
        if finish_reason == "length" or (drop_missing_finish_reason and finish_reason is None):
            dropped.append(record)
            continue
        kept.append(record)

    return kept, dropped


def missing_trial_counts_by_task(
    records: Sequence[dict[str, Any]],
    *,
    target_trials: int,
    task_ids: Sequence[str],
) -> dict[str, int]:
    """Compute remaining trial counts required to reach the target per task."""

    completed = Counter(str(record["task_id"]) for record in records)
    deficits: dict[str, int] = {}
    for task_id in task_ids:
        missing = target_trials - completed[str(task_id)]
        if missing > 0:
            deficits[str(task_id)] = missing
    return deficits


def write_generation_records_jsonl(records: Iterable[dict[str, Any]], output_path: str | Path) -> Path:
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return destination


def write_task_split_records(records: Sequence[dict[str, Any]], tasks_dir: str | Path) -> list[Path]:
    """Rewrite per-task JSONL files for a curated generation dataset."""

    destination = Path(tasks_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_task[str(record["task_id"])].append(record)

    written_paths: list[Path] = []
    for task_id, task_records in sorted(by_task.items()):
        task_records = sorted(task_records, key=lambda row: int(row.get("trial_index", 0)))
        task_path = destination / f"{task_id}.jsonl"
        with task_path.open("w", encoding="utf-8") as handle:
            for record in task_records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        written_paths.append(task_path)

    return written_paths
