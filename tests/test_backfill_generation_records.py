from __future__ import annotations

import importlib.util
import json
import subprocess
import tempfile
from pathlib import Path
import unittest
from unittest import mock

from qasserbench.generation.curation import missing_trial_counts_by_task


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "backfill_generation_records.py"
SPEC = importlib.util.spec_from_file_location("backfill_generation_records", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _record(task_id: str, trial_index: int) -> dict[str, object]:
    return {
        "model_id": "gemini-3.1-pro-preview",
        "provider_model_id": "gemini-3.1-pro-preview",
        "task_id": task_id,
        "trial_index": trial_index,
        "task_family": "demo",
        "property_type": "demo",
        "manifest_path": f"/tmp/{task_id}/task.yaml",
        "prompt_version": "test",
        "raw_response": "```python\nassert True\n```",
        "raw_payload": {"trial_index": trial_index, "finish_reason": "STOP"},
    }


class BackfillResumeTest(unittest.TestCase):
    def test_missing_trial_counts_deduplicates_same_trial_seen_in_multiple_sources(self) -> None:
        records = [
            _record("QAB02", index)
            for index in range(1, 18)
        ] + [
            _record("QAB02", index)
            for index in range(1, 18)
        ]

        deficits = missing_trial_counts_by_task(
            records,
            target_trials=20,
            task_ids=["QAB02"],
        )

        self.assertEqual(deficits, {"QAB02": 3})

    def test_backfill_appends_directly_to_task_file_with_next_trial_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            tasks_root = tmp / "tasks_root"
            model_dir = tmp / "model_dir"
            tasks_dir = model_dir / "tasks"

            for task_id in ("QAB01", "QAB02"):
                task_dir = tasks_root / task_id.lower()
                task_dir.mkdir(parents=True, exist_ok=True)
                (task_dir / "task.yaml").write_text(
                    f"task_id: {task_id}\n",
                    encoding="utf-8",
                )

            _write_jsonl(tasks_dir / "QAB01.jsonl", [_record("QAB01", i) for i in range(1, 21)])
            _write_jsonl(tasks_dir / "QAB02.jsonl", [_record("QAB02", i) for i in range(1, 4)])

            def fake_run(command: list[str], cwd: Path, text: bool, capture_output: bool) -> subprocess.CompletedProcess[str]:
                self.assertIn("--task-id", command)
                self.assertEqual(command[command.index("--task-id") + 1], "QAB02")
                self.assertEqual(command[command.index("--trials") + 1], "17")
                self.assertEqual(command[command.index("--trial-start-index") + 1], "4")
                self.assertEqual(command[command.index("--request-timeout-seconds") + 1], "120.0")
                self.assertIn("--append", command)
                output_path = Path(command[2])
                self.assertEqual(output_path.name, "QAB02.jsonl")
                with output_path.open("a", encoding="utf-8") as handle:
                    for row in [_record("QAB02", i) for i in range(4, 21)]:
                        handle.write(json.dumps(row, sort_keys=True) + "\n")
                return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

            with mock.patch.object(MODULE.subprocess, "run", side_effect=fake_run):
                MODULE.backfill_model_directory(
                    model_dir=model_dir,
                    model_name="gemini-3.1-pro-preview",
                    tasks_root=tasks_root,
                    client_type="gemini-native",
                    api_key_env="Gemini_API_KEY",
                    api_base_url="https://generativelanguage.googleapis.com",
                    anthropic_version="2023-06-01",
                    target_trials=20,
                    temperature=1.0,
                    max_output_tokens=2048,
                    request_timeout_seconds=120.0,
                    concurrency=1,
                    supplement_tag="resume_test",
                )

            task_rows = MODULE.read_generation_records(tasks_dir / "QAB02.jsonl")
            self.assertEqual(len(task_rows), 20)
            self.assertEqual([row["trial_index"] for row in task_rows], list(range(1, 21)))
            combined_rows = MODULE.read_generation_records(model_dir / "generation_records.jsonl")
            self.assertEqual(len(combined_rows), 40)

    def test_backfill_uses_existing_task_file_as_single_source_of_truth(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            tasks_root = tmp / "tasks_root"
            model_dir = tmp / "model_dir"
            tasks_dir = model_dir / "tasks"

            task_dir = tasks_root / "qab09"
            task_dir.mkdir(parents=True, exist_ok=True)
            (task_dir / "task.yaml").write_text(
                "task_id: QAB09\n",
                encoding="utf-8",
            )

            _write_jsonl(tasks_dir / "QAB09.jsonl", [_record("QAB09", i) for i in range(1, 15)])

            def fake_run(command: list[str], cwd: Path, text: bool, capture_output: bool) -> subprocess.CompletedProcess[str]:
                self.assertEqual(command[command.index("--task-id") + 1], "QAB09")
                self.assertEqual(command[command.index("--trials") + 1], "6")
                self.assertEqual(command[command.index("--trial-start-index") + 1], "15")
                output_path = Path(command[2])
                self.assertEqual(output_path, tasks_dir / "QAB09.jsonl")
                with output_path.open("a", encoding="utf-8") as handle:
                    for row in [_record("QAB09", i) for i in range(15, 21)]:
                        handle.write(json.dumps(row, sort_keys=True) + "\n")
                return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

            with mock.patch.object(MODULE.subprocess, "run", side_effect=fake_run):
                MODULE.backfill_model_directory(
                    model_dir=model_dir,
                    model_name="gemini-2.5-flash",
                    tasks_root=tasks_root,
                    client_type="gemini-native",
                    api_key_env="Gemini_API_KEY",
                    api_base_url="https://generativelanguage.googleapis.com",
                    anthropic_version="2023-06-01",
                    target_trials=20,
                    temperature=1.0,
                    max_output_tokens=0,
                    request_timeout_seconds=120.0,
                    concurrency=5,
                    supplement_tag="resume_test",
                )

            task_rows = MODULE.read_generation_records(tasks_dir / "QAB09.jsonl")
            self.assertEqual(len(task_rows), 20)
            self.assertEqual([row["trial_index"] for row in task_rows], list(range(1, 21)))


if __name__ == "__main__":
    unittest.main()
