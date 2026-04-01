from pathlib import Path
import unittest

from qasserbench.benchmark.loader import load_task_assets
from qasserbench.execution.interfaces import ExecutionConfig
from qasserbench.generation.prompting import inspect_task_prompt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASKS_ROOT = PROJECT_ROOT / "benchmark_data" / "tasks"
TASK_IDS = ("qab35", "qab36", "qab37")


class QAB3537QualityTests(unittest.TestCase):
    def test_prompts_start_with_program_summary_and_request_assertion_snippet(self) -> None:
        for task_id in TASK_IDS:
            with self.subTest(task_id=task_id):
                prompt_text = (TASKS_ROOT / task_id / "prompt.md").read_text(encoding="utf-8").strip()
                first_line = prompt_text.splitlines()[0].strip()
                lowered = prompt_text.lower()

                self.assertTrue(first_line.startswith("This program") or first_line.startswith("This circuit"))
                self.assertIn("assertion snippet", lowered)
                self.assertNotIn("complete modified program", lowered)

        self.assertNotIn(
            "even-parity",
            (TASKS_ROOT / "qab35" / "prompt.md").read_text(encoding="utf-8").lower(),
        )
        self.assertNotIn(
            "xor",
            (TASKS_ROOT / "qab36" / "prompt.md").read_text(encoding="utf-8").lower(),
        )
        self.assertNotIn(
            "single-excitation",
            (TASKS_ROOT / "qab37" / "prompt.md").read_text(encoding="utf-8").lower(),
        )

    def test_gold_metadata_required_terms_exist_in_gold_source(self) -> None:
        for task_id in TASK_IDS:
            with self.subTest(task_id=task_id):
                assets = load_task_assets(TASKS_ROOT / task_id / "task.yaml")
                required_terms = tuple(assets.task.gold_metadata.get("required_substrings", ()))

                self.assertTrue(required_terms)
                for term in required_terms:
                    self.assertIn(term, assets.gold_source)

    def test_prompt_rendering_includes_build_and_run_paths(self) -> None:
        for task_id in TASK_IDS:
            with self.subTest(task_id=task_id):
                assets = load_task_assets(TASKS_ROOT / task_id / "task.yaml")
                context = inspect_task_prompt(assets)

                self.assertIn("def build_circuit(", context.source_excerpt)
                self.assertIn("def run_program(", context.source_excerpt)
                self.assertGreaterEqual(len(context.selected_function_names), 2)

        qab36_context = inspect_task_prompt(load_task_assets(TASKS_ROOT / "qab36" / "task.yaml"))
        self.assertNotIn("XOR_MASK", qab36_context.source_excerpt)

        qab37_context = inspect_task_prompt(load_task_assets(TASKS_ROOT / "qab37" / "task.yaml"))
        self.assertNotIn("LEGAL_STATES", qab37_context.source_excerpt)

    def test_runtime_metadata_does_not_leak_family_or_mask_hints(self) -> None:
        expected_absent_keys = {
            "qab35": {"support_family"},
            "qab36": {"xor_mask", "register_width"},
            "qab37": {"support_family", "legal_states"},
        }

        for task_id, absent_keys in expected_absent_keys.items():
            with self.subTest(task_id=task_id):
                assets = load_task_assets(TASKS_ROOT / task_id / "task.yaml")
                config = ExecutionConfig(
                    shots=assets.task.shots,
                    backend="aer_simulator",
                    seed=123,
                    metadata={},
                )

                nominal = assets.program.run_program(config)
                for key in absent_keys:
                    self.assertNotIn(key, nominal.metadata)

    def test_gold_evaluator_passes_nominal_and_rejects_faults(self) -> None:
        for task_id in TASK_IDS:
            with self.subTest(task_id=task_id):
                assets = load_task_assets(TASKS_ROOT / task_id / "task.yaml")
                config = ExecutionConfig(
                    shots=assets.task.shots,
                    backend="aer_simulator",
                    seed=123,
                    metadata={},
                )

                nominal = assets.program.run_program(config)
                nominal_gold = assets.program.evaluate_gold_assertion(nominal)
                self.assertTrue(nominal_gold.passed, task_id)

                for fault_id, fault_program in assets.fault_programs.items():
                    fault_result = fault_program.run_program(config)
                    fault_gold = assets.program.evaluate_gold_assertion(fault_result)
                    self.assertFalse(fault_gold.passed, f"{task_id}:{fault_id}")

    def test_qab35_detects_odd_parity_leakage(self) -> None:
        assets = load_task_assets(TASKS_ROOT / "qab35" / "task.yaml")
        config = ExecutionConfig(shots=assets.task.shots, backend="aer_simulator", seed=11, metadata={})

        nominal = assets.program.run_program(config)
        self.assertTrue(all(bitstring.count("1") % 2 == 0 for bitstring in nominal.counts))

        faulty = assets.fault_programs["odd_parity_leakage"].run_program(config)
        self.assertTrue(any(bitstring.count("1") % 2 == 1 for bitstring in faulty.counts))

    def test_qab36_requires_cross_register_relation(self) -> None:
        assets = load_task_assets(TASKS_ROOT / "qab36" / "task.yaml")
        config = ExecutionConfig(shots=assets.task.shots, backend="aer_simulator", seed=17, metadata={})

        nominal = assets.program.run_program(config)
        register_width = len(next(iter(nominal.counts))) // 2
        observed_masks = set()

        for bitstring in nominal.counts:
            output_bits = bitstring[:-register_width]
            input_bits = bitstring[-register_width:]
            observed_mask = "".join(
                str(int(left_bit) ^ int(right_bit))
                for left_bit, right_bit in zip(output_bits, input_bits)
            )
            observed_masks.add(observed_mask)

        self.assertEqual(observed_masks, {"101"})

    def test_qab37_family_spreads_mass_across_legal_states(self) -> None:
        assets = load_task_assets(TASKS_ROOT / "qab37" / "task.yaml")
        config = ExecutionConfig(shots=assets.task.shots, backend="aer_simulator", seed=23, metadata={})

        nominal = assets.program.run_program(config)
        legal_states = {
            bitstring
            for bitstring in nominal.counts
            if bitstring.count("1") == 1
        }
        self.assertGreaterEqual(len(legal_states), 4)
        self.assertLess(max(nominal.counts.values()) / nominal.shots, 0.5)

        faulty = assets.fault_programs["collapsed_excitation_family"].run_program(config)
        self.assertEqual(len(faulty.counts), 1)


if __name__ == "__main__":
    unittest.main()
