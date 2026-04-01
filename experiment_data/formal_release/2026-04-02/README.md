# Formal Experiment Release (2026-04-02)

This directory snapshots the formal experiment artifacts that existed in the workspace on 2026-04-02.

Included generated-instance runs:

- `generated_instances/openrouter-main/deepseek_deepseek-v3.2`
  - status: complete
  - task coverage: `37/37`
  - total records: `740`
- `generated_instances/openrouter-main/qwen_qwen3.5-397b-a17b`
  - status: complete
  - task coverage: `37/37`
  - total records: `740`
- `generated_instances/openrouter-main/meta-llama_llama-4-scout`
  - status: complete
  - task coverage: `37/37`
  - total records: `740`
- `generated_instances/openai-main/gpt-5.4`
  - status: complete
  - task coverage: `37/37`
  - total records: `740`
- `generated_instances/anthropic-main/claude-sonnet-4-20250514`
  - status: complete
  - task coverage: `37/37`
  - total records: `740`
- `generated_instances/gemini-main/gemini-3.1-flash-lite-preview`
  - status: partial / in progress when snapped
  - task coverage: `7/37`
  - total records: `140`

Included evaluation artifacts:

- `raw_results/openai-main/gpt-5.4/trial_results.jsonl`
- `raw_results/openrouter-main/qwen_qwen3.5-397b-a17b/trial_results.jsonl`
- `summaries/openai-main/gpt-5.4/summary.json`

Notes:

- This release intentionally excludes smoke tests, token probes, and precheck runs.
- The Gemini run was preserved as-is at the time of export; it was not complete.
- The Gemini snapshot still contains `_corrupt_backup_20260402/` and `_resume_tmp/` because those were part of the live recovery state when this release was created.
- The release is a tracked copy of selected artifacts. The original working directories under `experiment_data/generated_instances`, `experiment_data/raw_results`, and `experiment_data/summaries` were left in place.
