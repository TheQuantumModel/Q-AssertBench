import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from qasserbench.generation.curation import (
    curate_generation_records,
    missing_trial_counts_by_task,
    write_task_split_records,
)


class GenerationCurationTests(unittest.TestCase):
    def test_curate_generation_records_drops_length_and_missing_finish_reason(self) -> None:
        records = [
            {
                "task_id": "QAB01",
                "trial_index": 1,
                "raw_payload": {"finish_reason": "stop"},
            },
            {
                "task_id": "QAB01",
                "trial_index": 2,
                "raw_payload": {"finish_reason": "length"},
            },
            {
                "task_id": "QAB01",
                "trial_index": 3,
                "raw_payload": {"finish_reason": None},
            },
            {
                "task_id": "QAB02",
                "trial_index": 1,
                "raw_payload": {"finish_reason": "stop"},
            },
        ]

        kept, dropped = curate_generation_records(records)

        self.assertEqual([(row["task_id"], row["trial_index"]) for row in kept], [("QAB01", 1), ("QAB02", 1)])
        self.assertEqual(
            [(row["task_id"], row["trial_index"]) for row in dropped],
            [("QAB01", 2), ("QAB01", 3)],
        )

    def test_missing_trial_counts_reflect_curated_records(self) -> None:
        curated = [
            {"task_id": "QAB01", "trial_index": 1, "raw_payload": {"finish_reason": "stop"}},
            {"task_id": "QAB01", "trial_index": 2, "raw_payload": {"finish_reason": "stop"}},
            {"task_id": "QAB02", "trial_index": 1, "raw_payload": {"finish_reason": "stop"}},
        ]

        missing = missing_trial_counts_by_task(
            curated,
            target_trials=3,
            task_ids=["QAB01", "QAB02", "QAB03"],
        )

        self.assertEqual(missing, {"QAB01": 1, "QAB02": 2, "QAB03": 3})

    def test_write_task_split_records_emits_one_jsonl_per_task(self) -> None:
        curated = [
            {"task_id": "QAB01", "trial_index": 2, "raw_payload": {"finish_reason": "stop"}},
            {"task_id": "QAB01", "trial_index": 1, "raw_payload": {"finish_reason": "stop"}},
            {"task_id": "QAB02", "trial_index": 1, "raw_payload": {"finish_reason": "stop"}},
        ]

        with TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            written = write_task_split_records(curated, output_dir)

            self.assertEqual(sorted(path.name for path in written), ["QAB01.jsonl", "QAB02.jsonl"])
            qab01_records = [
                json.loads(line)
                for line in (output_dir / "QAB01.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual([row["trial_index"] for row in qab01_records], [1, 2])


if __name__ == "__main__":
    unittest.main()
