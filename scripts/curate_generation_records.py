"""Backup raw generation files and rewrite curated generation datasets."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from qasserbench.generation.curation import (
    curate_generation_records,
    missing_trial_counts_by_task,
    write_generation_records_jsonl,
    write_task_split_records,
)
from qasserbench.generation.driver import read_generation_records


def _copy_if_exists(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)
        return
    shutil.copy2(source, destination)


def curate_model_directory(
    *,
    model_dir: str | Path,
    backup_root: str | Path,
    task_ids: list[str],
    target_trials: int,
) -> dict[str, object]:
    source_dir = Path(model_dir).resolve()
    backup_dir = Path(backup_root).resolve() / source_dir.name
    combined_path = source_dir / "generation_records.jsonl"
    tasks_dir = source_dir / "tasks"

    _copy_if_exists(combined_path, backup_dir / "generation_records.jsonl")
    _copy_if_exists(tasks_dir, backup_dir / "tasks")

    original_records = read_generation_records(combined_path)
    curated_records, dropped_records = curate_generation_records(original_records)

    write_generation_records_jsonl(curated_records, combined_path)
    write_task_split_records(curated_records, tasks_dir)

    missing = missing_trial_counts_by_task(
        curated_records,
        target_trials=target_trials,
        task_ids=task_ids,
    )

    return {
        "model_dir": str(source_dir),
        "backup_dir": str(backup_dir),
        "original_count": len(original_records),
        "curated_count": len(curated_records),
        "dropped_count": len(dropped_records),
        "missing": missing,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Backup raw generation files and curate records.")
    parser.add_argument(
        "--model-dir",
        dest="model_dirs",
        action="append",
        required=True,
        help="Model generation directory containing generation_records.jsonl",
    )
    parser.add_argument(
        "--backup-root",
        type=Path,
        required=True,
        help="Directory that stores untouched raw generation backups",
    )
    parser.add_argument(
        "--task-id",
        dest="task_ids",
        action="append",
        required=True,
        help="Benchmark task id; may be repeated",
    )
    parser.add_argument(
        "--target-trials",
        type=int,
        default=20,
        help="Desired number of retained generation records per task",
    )
    args = parser.parse_args()

    for model_dir in args.model_dirs:
        summary = curate_model_directory(
            model_dir=model_dir,
            backup_root=args.backup_root,
            task_ids=list(args.task_ids),
            target_trials=args.target_trials,
        )
        print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
