import unittest

from qasserbench.execution.interfaces import ExecutionConfig, ExecutionResult, ProgramDefinition
from qasserbench.execution.runner import run_candidate_trial
from qasserbench.generation.extract import extract_candidate_assertion


def _make_program_definition(task_id: str, counts: dict[str, int]) -> ProgramDefinition:
    def build_program() -> str:
        return task_id

    def run_program(config: ExecutionConfig) -> ExecutionResult:
        return ExecutionResult(
            counts=dict(counts),
            shots=config.shots,
            backend=config.backend,
            metadata={"task_id": task_id},
        )

    return ProgramDefinition(
        task_id=task_id,
        build_program=build_program,
        run_program=run_program,
        evaluate_gold_assertion=lambda result: None,  # not used in Task 5
    )


class ExecutionRunnerTests(unittest.TestCase):
    def test_runs_candidate_on_nominal_and_fault_variants(self) -> None:
        nominal_program = _make_program_definition("QAB01", {"00": 5, "11": 1})
        fault_programs = {
            "missing_dominant_zero": _make_program_definition("QAB01_fault", {"00": 1, "11": 5}),
        }
        artifact = extract_candidate_assertion(
            'assert counts["00"] > counts["11"]',
            extraction_mode="assertion_block",
        )

        trial = run_candidate_trial(
            program=nominal_program,
            fault_programs=fault_programs,
            artifact=artifact,
            config=ExecutionConfig(shots=6, backend="test-backend"),
        )

        self.assertTrue(trial.nominal_assertion.passed)
        self.assertIsNone(trial.nominal_assertion.error_type)
        self.assertEqual(len(trial.fault_results), 1)
        self.assertEqual(trial.fault_results[0].fault_id, "missing_dominant_zero")
        self.assertFalse(trial.fault_results[0].assertion_result.passed)
        self.assertEqual(trial.fault_results[0].assertion_result.error_type, "assertion_failed")

    def test_marks_runtime_errors_from_candidate_execution(self) -> None:
        nominal_program = _make_program_definition("QAB01", {"00": 5, "11": 1})
        artifact = extract_candidate_assertion(
            "raise RuntimeError('boom')",
            extraction_mode="assertion_block",
        )

        trial = run_candidate_trial(
            program=nominal_program,
            fault_programs={},
            artifact=artifact,
            config=ExecutionConfig(shots=6, backend="test-backend"),
        )

        self.assertFalse(trial.nominal_assertion.passed)
        self.assertEqual(trial.nominal_assertion.error_type, "runtime_error")
        self.assertIn("boom", trial.nominal_assertion.message or "")

    def test_allows_generator_expressions_to_reference_prior_local_bindings(self) -> None:
        nominal_program = _make_program_definition("QAB35", {"0000": 4, "0011": 2})
        artifact = extract_candidate_assertion(
            "legal_outcomes = {'0000', '0011'}\n"
            "observed = set(counts.keys())\n"
            "assert all(outcome in legal_outcomes for outcome in observed)",
            extraction_mode="assertion_block",
        )

        trial = run_candidate_trial(
            program=nominal_program,
            fault_programs={},
            artifact=artifact,
            config=ExecutionConfig(shots=6, backend="test-backend"),
        )

        self.assertTrue(trial.nominal_assertion.passed)
        self.assertIsNone(trial.nominal_assertion.error_type)


if __name__ == "__main__":
    unittest.main()
