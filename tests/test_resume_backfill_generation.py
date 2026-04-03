from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
import unittest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "resume_backfill_generation.py"


def _load_module():
    assert SCRIPT_PATH.exists(), f"missing script: {SCRIPT_PATH}"
    spec = importlib.util.spec_from_file_location("resume_backfill_generation", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ResumeBackfillGenerationTests(unittest.TestCase):
    def test_resume_retries_failed_backfill_after_delay_until_success(self) -> None:
        module = _load_module()

        calls: list[list[str]] = []
        sleeps: list[int] = []
        logs: list[str] = []

        results = iter(
            [
                subprocess.CompletedProcess(["python", "backfill"], 1),
                subprocess.CompletedProcess(["python", "backfill"], 0),
            ]
        )

        def fake_runner(command: list[str], *, cwd: Path, text: bool, check: bool) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            return next(results)

        def fake_sleep(seconds: int) -> None:
            sleeps.append(seconds)

        exit_code = module.resume_backfill_until_complete(
            ["python", "backfill"],
            cwd=Path("/tmp"),
            retry_delay_seconds=60,
            runner=fake_runner,
            sleeper=fake_sleep,
            log=logs.append,
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(calls, [["python", "backfill"], ["python", "backfill"]])
        self.assertEqual(sleeps, [60])
        self.assertTrue(any("retrying in 60s" in entry for entry in logs))

    def test_build_backfill_command_preserves_expected_arguments(self) -> None:
        module = _load_module()

        command = module.build_backfill_command(
            python_bin=Path("/tmp/.venv/bin/python"),
            backfill_script=Path("/repo/scripts/backfill_generation_records.py"),
            model_dir=Path("/tmp/model_dir"),
            model_name="gemini-3-flash-preview",
            tasks_root=Path("/tmp/tasks_root"),
            client_type="gemini-native",
            api_key_env="Gemini_API_KEY",
            api_base_url=None,
            anthropic_version="2023-06-01",
            target_trials=20,
            temperature=1.0,
            max_output_tokens=0,
            request_timeout_seconds=120.0,
            concurrency=1,
            supplement_tag="unused",
        )

        self.assertEqual(command[0], "/tmp/.venv/bin/python")
        self.assertEqual(command[1], "/repo/scripts/backfill_generation_records.py")
        self.assertIn("--model-dir", command)
        self.assertIn("/tmp/model_dir", command)
        self.assertIn("--model-name", command)
        self.assertIn("gemini-3-flash-preview", command)
        self.assertIn("--client", command)
        self.assertIn("gemini-native", command)
        self.assertIn("--api-key-env", command)
        self.assertIn("Gemini_API_KEY", command)
        self.assertIn("--max-output-tokens", command)
        self.assertIn("0", command)
        self.assertIn("--request-timeout-seconds", command)
        self.assertIn("120.0", command)
        self.assertIn("--concurrency", command)
        self.assertIn("1", command)


if __name__ == "__main__":
    unittest.main()
