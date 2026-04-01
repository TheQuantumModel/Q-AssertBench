from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

from qasserbench.benchmark.loader import load_task_assets
from qasserbench.generation.clients import GenerationResponse, ModelClient
from qasserbench.reporting.io import read_trial_results
from scripts.run_generation import run_generation_experiment
from scripts.run_evaluation import evaluate_generation_records
from scripts.summarize_results import summarize


class GoldReplayClient(ModelClient):
    def __init__(self, task_to_response: dict[str, str]) -> None:
        self._task_to_response = dict(task_to_response)

    @property
    def model_id(self) -> str:
        return "gold-replay"

    def generate(self, *, prompt_text: str, task_id: str, trial_index: int) -> GenerationResponse:
        return GenerationResponse(
            text=self._task_to_response[task_id],
            payload={
                "task_id": task_id,
                "trial_index": trial_index,
                "prompt_length": len(prompt_text),
            },
        )


class EndToEndPilotTests(unittest.TestCase):
    def test_runs_generation_evaluation_and_summary_for_three_pilot_tasks(self) -> None:
        tasks_root = Path("/home/li/project/Q-AssertBench/project_code/benchmark_data/tasks")
        task_ids = ("QAB01", "QAB10", "QAB20")
        task_to_response: dict[str, str] = {}

        for task_id in task_ids:
            manifest_path = tasks_root / task_id.lower() / "task.yaml"
            assets = load_task_assets(manifest_path)
            task_to_response[task_id] = f"```python\n{assets.gold_source}\n```"

        client = GoldReplayClient(task_to_response)

        with TemporaryDirectory() as tmp_dir:
            experiment_root = Path(tmp_dir) / "experiment_data"
            generation_path = experiment_root / "generated" / "pilot_generation.jsonl"
            trial_results_path = experiment_root / "raw_results" / "pilot_trial_results.jsonl"
            summary_path = experiment_root / "summaries" / "pilot_summary.json"

            run_generation_experiment(
                tasks_root=tasks_root,
                output_path=generation_path,
                client=client,
                task_ids=task_ids,
                trial_count=2,
            )
            evaluate_generation_records(
                input_path=generation_path,
                output_path=trial_results_path,
            )
            summarize(str(trial_results_path), str(summary_path), pass_k_values=(1, 2))

            trial_records = read_trial_results(trial_results_path)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(len(trial_records), 6)
        self.assertEqual({record["task_id"] for record in trial_records}, set(task_ids))
        self.assertTrue(all(record["outcome"] == "pass" for record in trial_records))
        self.assertTrue(all(record["overall_outcome"] == "pass" for record in trial_records))
        self.assertTrue(all(record["generation_status"] == "success" for record in trial_records))
        self.assertTrue(all(record["nominal_status"] == "pass" for record in trial_records))
        self.assertTrue(all(record["fault_status"] == "detected" for record in trial_records))
        self.assertTrue(all(record["alignment_label"] == "aligned" for record in trial_records))
        self.assertTrue(all("alignment_components" in record for record in trial_records))
        self.assertTrue(
            all(record["alignment_components"]["agreement_score"] == 1.0 for record in trial_records)
        )
        self.assertTrue(all(record["gold_nominal_passed"] for record in trial_records))
        self.assertTrue(all("generation_temperature" in record for record in trial_records))
        self.assertTrue(all("generation_max_output_tokens" in record for record in trial_records))
        self.assertTrue(all("prompt_tokens" in record for record in trial_records))
        self.assertTrue(all("completion_tokens" in record for record in trial_records))
        self.assertTrue(all("total_tokens" in record for record in trial_records))

        model_summary = next(
            item for item in summary["model_summaries"] if item["model_id"] == "gold-replay"
        )
        self.assertEqual(model_summary["trial_count"], 6)
        self.assertAlmostEqual(model_summary["pass_rate"], 1.0)
        self.assertAlmostEqual(model_summary["pass@1"], 1.0)
        self.assertAlmostEqual(model_summary["pass@2"], 1.0)

    def test_records_structured_status_fields_for_runtime_error_candidate(self) -> None:
        tasks_root = Path("/home/li/project/Q-AssertBench/project_code/benchmark_data/tasks")
        client = GoldReplayClient({"QAB01": "```python\nraise RuntimeError('boom')\n```"})

        with TemporaryDirectory() as tmp_dir:
            experiment_root = Path(tmp_dir) / "experiment_data"
            generation_path = experiment_root / "generated" / "runtime_generation.jsonl"
            trial_results_path = experiment_root / "raw_results" / "runtime_trial_results.jsonl"

            run_generation_experiment(
                tasks_root=tasks_root,
                output_path=generation_path,
                client=client,
                task_ids=("QAB01",),
                trial_count=1,
            )
            evaluate_generation_records(
                input_path=generation_path,
                output_path=trial_results_path,
            )

            trial_records = read_trial_results(trial_results_path)

        self.assertEqual(len(trial_records), 1)
        record = trial_records[0]
        self.assertEqual(record["generation_status"], "success")
        self.assertEqual(record["nominal_status"], "runtime_error")
        self.assertEqual(record["fault_status"], "runtime_error")
        self.assertEqual(record["overall_outcome"], "invalid")
        self.assertEqual(record["outcome"], "invalid")


if __name__ == "__main__":
    unittest.main()
