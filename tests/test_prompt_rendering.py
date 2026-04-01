from pathlib import Path
import unittest

from qasserbench.benchmark.loader import load_task_assets
from qasserbench.generation.prompting import (
    COMMON_PROMPT_INTRO,
    PROMPT_TEMPLATE_VERSION,
    render_task_prompt,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASKS_ROOT = PROJECT_ROOT / "benchmark_data" / "tasks"


class PromptRenderingTests(unittest.TestCase):
    def test_raw_task_prompts_focus_on_task_semantics_not_output_contract(self) -> None:
        manifest_paths = sorted(TASKS_ROOT.glob("*/task.yaml"))

        for manifest_path in manifest_paths:
            with self.subTest(task=manifest_path.parent.name):
                prompt_text = (manifest_path.parent / "prompt.md").read_text(encoding="utf-8")
                self.assertNotIn("Output only the assertion snippet.", prompt_text)
                self.assertNotIn("No explanations, no comments.", prompt_text)

    def test_renders_standard_task_with_common_contract_and_source_excerpt(self) -> None:
        assets = load_task_assets(TASKS_ROOT / "qab01" / "task.yaml")

        prompt_text = render_task_prompt(assets)

        self.assertIn(COMMON_PROMPT_INTRO, prompt_text)
        self.assertIn("Return one Python code block containing only the assertion snippet.", prompt_text)
        self.assertIn("write their values directly in the snippet", prompt_text)
        self.assertIn("Relevant source excerpt from `program.py`:", prompt_text)
        self.assertIn("```python", prompt_text)
        self.assertIn("def build_circuit", prompt_text)
        self.assertIn("def run_program", prompt_text)
        self.assertIn("Prompt template version", prompt_text)
        self.assertIn(PROMPT_TEMPLATE_VERSION, prompt_text)

    def test_renders_probe_task_with_probe_function_and_observation_stage(self) -> None:
        assets = load_task_assets(TASKS_ROOT / "qab15" / "task.yaml")

        prompt_text = render_task_prompt(assets)

        self.assertIn("after_state_preparation", prompt_text)
        self.assertIn("build_state_preparation_probe_circuit", prompt_text)
        self.assertIn("The provided `counts` are produced by the probe function", prompt_text)
        self.assertIn("The snippet itself runs with `counts`, `shots`, and `metadata`", prompt_text)
        self.assertIn("def run_program", prompt_text)

    def test_renders_all_tasks_with_common_contract_and_source_block(self) -> None:
        manifest_paths = sorted(TASKS_ROOT.glob("*/task.yaml"))

        for manifest_path in manifest_paths:
            with self.subTest(task=manifest_path.parent.name):
                assets = load_task_assets(manifest_path)
                prompt_text = render_task_prompt(assets)

                self.assertIn(COMMON_PROMPT_INTRO, prompt_text)
                self.assertIn("Relevant source excerpt from `program.py`:", prompt_text)
                self.assertIn("```python", prompt_text)
                self.assertIn("def run_program", prompt_text)


if __name__ == "__main__":
    unittest.main()
