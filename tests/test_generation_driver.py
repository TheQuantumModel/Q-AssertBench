from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import time
import unittest
import yaml

from qasserbench.generation.clients import GenerationResponse, ModelClient
from qasserbench.generation.driver import (
    discover_task_manifests,
    read_generation_records,
    run_generation_trials,
)
from qasserbench.generation.prompting import COMMON_PROMPT_INTRO, PROMPT_TEMPLATE_VERSION
from scripts.run_generation import (
    load_generation_manifest,
    resolve_manifest_task_ids,
    run_manifest_generation_experiment,
)


class FakeModelClient(ModelClient):
    def __init__(self, model_id: str) -> None:
        self._model_id = model_id

    @property
    def model_id(self) -> str:
        return self._model_id

    def generate(self, *, prompt_text: str, task_id: str, trial_index: int) -> GenerationResponse:
        return GenerationResponse(
            text=f"```python\nassert '{task_id}:{trial_index}'\n```",
            payload={
                "task_id": task_id,
                "trial_index": trial_index,
                "prompt_prefix": prompt_text.splitlines()[0],
            },
        )


class ConcurrentTrackingClient(ModelClient):
    def __init__(self, model_id: str, *, delay_seconds: float = 0.05) -> None:
        self._model_id = model_id
        self._delay_seconds = delay_seconds
        self._lock = threading.Lock()
        self.inflight = 0
        self.max_inflight = 0

    @property
    def model_id(self) -> str:
        return self._model_id

    def generate(self, *, prompt_text: str, task_id: str, trial_index: int) -> GenerationResponse:
        with self._lock:
            self.inflight += 1
            self.max_inflight = max(self.max_inflight, self.inflight)
        try:
            time.sleep(self._delay_seconds)
            return GenerationResponse(
                text=f"```python\nassert '{task_id}:{trial_index}'\n```",
                payload={
                    "task_id": task_id,
                    "trial_index": trial_index,
                },
            )
        finally:
            with self._lock:
                self.inflight -= 1


class GenerationDriverTests(unittest.TestCase):
    def test_writes_repeated_generation_records_with_raw_payload(self) -> None:
        tasks_root = Path("/home/li/project/Q-AssertBench/project_code/benchmark_data/tasks")
        manifest_paths = discover_task_manifests(tasks_root, task_ids=("QAB01", "QAB10"))
        client = FakeModelClient("fake-model")

        with TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "generation_records.jsonl"

            written_path = run_generation_trials(
                manifest_paths=manifest_paths,
                client=client,
                trial_count=2,
                output_path=output_path,
            )
            records = read_generation_records(written_path)

        self.assertEqual(written_path, output_path.resolve())
        self.assertEqual(len(records), 4)

        first = records[0]
        self.assertEqual(first["model_id"], "fake-model")
        self.assertEqual(first["task_id"], "QAB01")
        self.assertEqual(first["trial_index"], 1)
        self.assertIn("raw_response", first)
        self.assertIn("raw_payload", first)
        self.assertEqual(first["raw_payload"]["task_id"], "QAB01")
        self.assertEqual(first["raw_payload"]["prompt_prefix"], COMMON_PROMPT_INTRO)
        self.assertEqual(first["prompt_version"], PROMPT_TEMPLATE_VERSION)
        self.assertIsNone(first["generation_temperature"])
        self.assertIsNone(first["generation_max_output_tokens"])
        self.assertIsNone(first["prompt_tokens"])
        self.assertIsNone(first["completion_tokens"])
        self.assertIsNone(first["total_tokens"])

    def test_filters_tasks_by_requested_task_id(self) -> None:
        tasks_root = Path("/home/li/project/Q-AssertBench/project_code/benchmark_data/tasks")
        manifest_paths = discover_task_manifests(tasks_root, task_ids=("QAB20",))
        client = FakeModelClient("fake-model")

        with TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "generation_records.jsonl"
            records = read_generation_records(
                run_generation_trials(
                    manifest_paths=manifest_paths,
                    client=client,
                    trial_count=3,
                    output_path=output_path,
                )
            )

        self.assertEqual(len(records), 3)
        self.assertEqual({record["task_id"] for record in records}, {"QAB20"})
        self.assertEqual([record["trial_index"] for record in records], [1, 2, 3])

    def test_can_run_trials_concurrently_and_write_all_records(self) -> None:
        tasks_root = Path("/home/li/project/Q-AssertBench/project_code/benchmark_data/tasks")
        manifest_paths = discover_task_manifests(tasks_root, task_ids=("QAB01",))
        client = ConcurrentTrackingClient("concurrent-model")

        with TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "generation_records.jsonl"
            records = read_generation_records(
                run_generation_trials(
                    manifest_paths=manifest_paths,
                    client=client,
                    trial_count=4,
                    output_path=output_path,
                    max_concurrency=3,
                )
            )

        self.assertGreaterEqual(client.max_inflight, 2)
        self.assertEqual(len(records), 4)
        self.assertEqual({record["trial_index"] for record in records}, {1, 2, 3, 4})
        self.assertEqual({record["task_id"] for record in records}, {"QAB01"})

    def test_writes_partial_generation_records_before_concurrent_run_finishes(self) -> None:
        tasks_root = Path("/home/li/project/Q-AssertBench/project_code/benchmark_data/tasks")
        manifest_paths = discover_task_manifests(tasks_root, task_ids=("QAB01",))
        client = ConcurrentTrackingClient("concurrent-model", delay_seconds=0.15)

        with TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "generation_records.jsonl"
            errors: list[BaseException] = []

            def _run() -> None:
                try:
                    run_generation_trials(
                        manifest_paths=manifest_paths,
                        client=client,
                        trial_count=4,
                        output_path=output_path,
                        max_concurrency=2,
                    )
                except BaseException as exc:  # pragma: no cover - test helper
                    errors.append(exc)

            worker = threading.Thread(target=_run)
            worker.start()
            partial_seen = False
            deadline = time.time() + 3.0
            while time.time() < deadline and worker.is_alive():
                if output_path.exists():
                    with output_path.open("r", encoding="utf-8") as handle:
                        line_count = sum(1 for line in handle if line.strip())
                    if 0 < line_count < 4:
                        partial_seen = True
                        break
                time.sleep(0.05)
            worker.join()

            if errors:
                raise errors[0]
            records = read_generation_records(output_path)

        self.assertTrue(partial_seen)
        self.assertEqual(len(records), 4)

    def test_manifest_include_mode_resolves_requested_tasks(self) -> None:
        tasks_root = Path("/home/li/project/Q-AssertBench/project_code/benchmark_data/tasks")

        with TemporaryDirectory() as tmp_dir:
            manifest_path = Path(tmp_dir) / "experiment.yaml"
            manifest_path.write_text(
                yaml.safe_dump(
                    {
                        "name": "include-test",
                        "client": "static",
                        "defaults": {
                            "tasks_root": str(tasks_root),
                            "output_dir": str(Path(tmp_dir) / "outputs"),
                            "trials": 2,
                            "response_text": "```python\nassert True\n```",
                        },
                        "task_selection": {
                            "mode": "include",
                            "task_ids": ["QAB01", "qab35"],
                        },
                        "models": [{"model_id": "static-a"}],
                    }
                ),
                encoding="utf-8",
            )

            manifest = load_generation_manifest(manifest_path)
            task_ids = resolve_manifest_task_ids(manifest)

        self.assertEqual(task_ids, ["QAB01", "QAB35"])

    def test_manifest_exclude_mode_omits_requested_tasks(self) -> None:
        tasks_root = Path("/home/li/project/Q-AssertBench/project_code/benchmark_data/tasks")

        with TemporaryDirectory() as tmp_dir:
            manifest_path = Path(tmp_dir) / "experiment.yaml"
            manifest_path.write_text(
                yaml.safe_dump(
                    {
                        "name": "exclude-test",
                        "client": "static",
                        "defaults": {
                            "tasks_root": str(tasks_root),
                            "output_dir": str(Path(tmp_dir) / "outputs"),
                            "trials": 1,
                            "response_text": "```python\nassert True\n```",
                        },
                        "task_selection": {
                            "mode": "exclude",
                            "task_ids": ["QAB01", "QAB02"],
                        },
                        "models": [{"model_id": "static-a"}],
                    }
                ),
                encoding="utf-8",
            )

            manifest = load_generation_manifest(manifest_path)
            task_ids = resolve_manifest_task_ids(manifest)

        self.assertNotIn("QAB01", task_ids)
        self.assertNotIn("QAB02", task_ids)
        self.assertIn("QAB03", task_ids)

    def test_manifest_batch_run_writes_one_output_per_model_and_honors_overrides(self) -> None:
        tasks_root = Path("/home/li/project/Q-AssertBench/project_code/benchmark_data/tasks")

        with TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "outputs"
            manifest_path = Path(tmp_dir) / "experiment.yaml"
            manifest_path.write_text(
                yaml.safe_dump(
                    {
                        "name": "batch-test",
                        "client": "static",
                        "defaults": {
                            "tasks_root": str(tasks_root),
                            "output_dir": str(output_dir),
                            "trials": 2,
                            "concurrency": 2,
                            "response_text": "```python\nassert True\n```",
                        },
                        "task_selection": {
                            "mode": "include",
                            "task_ids": ["QAB01"],
                        },
                        "models": [
                            {"model_id": "provider/model-a"},
                            {"model_id": "provider/model-b", "trials": 1},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            written_paths = run_manifest_generation_experiment(manifest_path)

            first_records = read_generation_records(written_paths[0])
            second_records = read_generation_records(written_paths[1])
            resolved_manifest_path = output_dir / "resolved_manifest.yaml"
            resolved_manifest_exists = resolved_manifest_path.exists()
            resolved_manifest = load_generation_manifest(resolved_manifest_path)

        self.assertEqual(len(written_paths), 2)
        self.assertEqual(len(first_records), 2)
        self.assertEqual(len(second_records), 1)
        self.assertEqual({record["model_id"] for record in first_records}, {"provider/model-a"})
        self.assertEqual({record["model_id"] for record in second_records}, {"provider/model-b"})
        self.assertTrue(resolved_manifest_exists)
        self.assertEqual(resolved_manifest["defaults"]["concurrency"], 2)

    def test_experiment_templates_exist_and_load(self) -> None:
        experiments_root = Path("/home/li/project/Q-AssertBench/project_code/experiments")
        template_paths = [
            experiments_root / "openrouter-main.yaml",
            experiments_root / "openrouter-supplement-temperature.yaml",
            experiments_root / "openrouter-supplement-full-vs-mini.yaml",
        ]

        for template_path in template_paths:
            self.assertTrue(template_path.exists(), f"Missing template: {template_path}")
            manifest = load_generation_manifest(template_path)
            self.assertIn("models", manifest)
            self.assertIn("defaults", manifest)

    def test_manifest_batch_run_supports_distinct_run_ids_for_same_provider_model(self) -> None:
        tasks_root = Path("/home/li/project/Q-AssertBench/project_code/benchmark_data/tasks")

        with TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "outputs"
            manifest_path = Path(tmp_dir) / "experiment.yaml"
            manifest_path.write_text(
                yaml.safe_dump(
                    {
                        "name": "temperature-test",
                        "client": "static",
                        "defaults": {
                            "tasks_root": str(tasks_root),
                            "output_dir": str(output_dir),
                            "trials": 1,
                            "response_text": "```python\nassert True\n```",
                        },
                        "task_selection": {
                            "mode": "include",
                            "task_ids": ["QAB01"],
                        },
                        "models": [
                            {"model_id": "provider/model-a", "run_id": "provider/model-a-temp1.0"},
                            {"model_id": "provider/model-a", "run_id": "provider/model-a-temp0.2"},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            written_paths = run_manifest_generation_experiment(manifest_path)
            first_records = read_generation_records(written_paths[0])
            second_records = read_generation_records(written_paths[1])

        self.assertEqual({record["model_id"] for record in first_records}, {"provider/model-a-temp1.0"})
        self.assertEqual({record["model_id"] for record in second_records}, {"provider/model-a-temp0.2"})


if __name__ == "__main__":
    unittest.main()
