# Q-AssertBench Project Code

This directory contains the runnable benchmark framework, task catalog, and execution-based evaluation pipeline for Q-AssertBench.

## Quick Start

### 1. Create the environment

This repository targets Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

### 2. Validate the benchmark tasks

```bash
python scripts/validate_tasks.py
```

### 3. Generate assertion candidates

```bash
cp examples/client_templates/openai-compatible.example.yaml /tmp/qab-client.yaml
export QAB_API_KEY="your-api-key"
python scripts/run_generation.py --manifest /tmp/qab-client.yaml
```

### 4. Evaluate generated results

```bash
python scripts/run_evaluation.py \
  path/to/generation_records.jsonl \
  path/to/trial_results.jsonl
```

### 5. Summarize trial-level results

```bash
python scripts/summarize_results.py \
  path/to/trial_results.jsonl \
  path/to/summary.json
```

### 6. Batch-evaluate a run directory

```bash
find experiment_data/generated_instances/YOUR_RUN -name generation_records.jsonl | while read -r gen; do
  model_dir=$(dirname "$gen")
  rel=${model_dir#experiment_data/generated_instances/}
  mkdir -p "experiment_data/raw_results/$rel" "experiment_data/summaries/$rel"

  python scripts/run_evaluation.py \
    "$gen" \
    "experiment_data/raw_results/$rel/trial_results.jsonl"

  python scripts/summarize_results.py \
    "experiment_data/raw_results/$rel/trial_results.jsonl" \
    "experiment_data/summaries/$rel/summary.json"
done
```

## Client Notes

The reference manifest at `examples/client_templates/openai-compatible.example.yaml` is intentionally provider-safe:

- copy it to a writable location and edit the provider-specific fields before running `run_generation.py`
- no API key is stored in the repository
- you must supply your own key through an environment variable such as `QAB_API_KEY`
- different providers may require different base URLs, model IDs, token limits, timeouts, or even a different client mode such as `anthropic-native` or `gemini-native`

The released evaluation pipeline is provider-agnostic once a `generation_records.jsonl` file has been produced. This scoring stage is the canonical path used for the released experiment data.

## Project Structure

- `benchmark_data/tasks/`: canonical task catalog, including prompts, gold assertions, and fault-injected counterparts
- `src/qasserbench/`: benchmark loader, generation clients, execution runtime, evaluation logic, and reporting utilities
- `tests/`: retained core tests for task loading and evaluation; internal recovery and provider-specific tests are intentionally omitted from the public artifact
- `scripts/run_generation.py`: repeated assertion generation for single-model or manifest-driven runs
- `scripts/run_evaluation.py`: execution-based evaluation from `generation_records.jsonl` to `trial_results.jsonl`
- `scripts/summarize_results.py`: summary aggregation from trial-level results
- `scripts/validate_tasks.py`: structural validation for the task catalog
- `examples/client_templates/`: provider-safe reference manifests with no embedded keys
- `experiments/`: experiment manifests used for the released study configurations
- `experiment_data/formal_release/2026-04-02/`: paper-aligned released generations, raw results, and summaries

## Project Summary

Q-AssertBench is a benchmark framework for studying assertion generation in quantum programs. The repository provides the task set, generation clients, execution-based evaluator, and reporting scripts needed to reproduce or extend the released experiments, with particular emphasis on the evaluation stage that turns raw generations into benchmark outcomes.
