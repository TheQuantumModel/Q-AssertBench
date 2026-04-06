# Q-AssertBench Project Code

This directory contains the runnable benchmark framework, task catalog, and execution-based evaluation pipeline for Q-AssertBench.

## Quick Start

### 1. Create the environment

This repository targets Python 3.10+.

On Debian/Ubuntu, you may need to install the system venv package first:

```bash
sudo apt install python3-venv
```

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m ensurepip --upgrade
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2. Validate the benchmark tasks

```bash
python scripts/validate_tasks.py
```

### 3. Generate raw model outputs with a client

For a single OpenAI-compatible provider, pass the client settings on the command line:

```bash
export QAB_API_KEY="your-api-key"

python scripts/run_generation.py \
  experiment_data/generated_instances/quickstart/single_model/generation_records.jsonl \
  --client openai-compatible \
  --tasks-root benchmark_data/tasks \
  --task-id QAB01 \
  --trials 2 \
  --concurrency 1 \
  --model-id your-provider/your-model \
  --model your-provider/your-model \
  --api-base-url https://your-provider.example/v1 \
  --api-key-env QAB_API_KEY \
  --temperature 1.0 \
  --max-output-tokens 2048
```

For batch generation, start from the reference manifest at `examples/client_templates/openai-compatible.example.yaml`:

```bash
cp examples/client_templates/openai-compatible.example.yaml /tmp/qab-client.yaml
# Edit base_url, output_dir, task_ids, and model_id entries for your provider.

export QAB_API_KEY="your-api-key"
python scripts/run_generation.py --manifest /tmp/qab-client.yaml
```

Important notes about client templates:

- No API key is stored in this repository.
- Template manifests are reference configurations only.
- Different providers may require different base URLs, model names, temperatures, token limits, timeouts, or even a different client mode such as `anthropic-native` or `gemini-native`.
- If you store credentials in `.env.local`, keep that file private and out of version control.

### 4. Evaluate generated results

For a single generated JSONL file:

```bash
python scripts/run_evaluation.py \
  experiment_data/generated_instances/quickstart/single_model/generation_records.jsonl \
  experiment_data/raw_results/quickstart/single_model/trial_results.jsonl

python scripts/summarize_results.py \
  experiment_data/raw_results/quickstart/single_model/trial_results.jsonl \
  experiment_data/summaries/quickstart/single_model/summary.json
```

For a batch run produced from a manifest, repeat the same evaluation pipeline for every generated `generation_records.jsonl`:

```bash
find experiment_data/generated_instances/quickstart-openai-compatible -name generation_records.jsonl | while read -r gen; do
  model_dir=$(dirname "$gen")
  rel=${model_dir#experiment_data/generated_instances/}

  mkdir -p "experiment_data/raw_results/$rel"
  mkdir -p "experiment_data/summaries/$rel"

  python scripts/run_evaluation.py \
    "$gen" \
    "experiment_data/raw_results/$rel/trial_results.jsonl"

  python scripts/summarize_results.py \
    "experiment_data/raw_results/$rel/trial_results.jsonl" \
    "experiment_data/summaries/$rel/summary.json"
done
```

The evaluation stage is the part used to score the released experiment data: raw model generations are executed against the nominal and faulty programs, classified into benchmark outcomes, and then aggregated into summary metrics.

## Project Structure

- `benchmark_data/tasks/`: canonical task catalog, including prompts, gold assertions, and fault-injected counterparts
- `src/qasserbench/`: benchmark loader, generation clients, execution runtime, evaluation logic, and reporting utilities
- `scripts/run_generation.py`: repeated assertion generation for single-model or manifest-driven runs
- `scripts/run_evaluation.py`: execution-based evaluation from `generation_records.jsonl` to `trial_results.jsonl`
- `scripts/summarize_results.py`: summary aggregation from trial-level results
- `scripts/validate_tasks.py`: structural validation for the task catalog
- `examples/client_templates/`: provider-safe reference manifests with no embedded keys
- `experiments/`: experiment manifests used for the released study configurations
- `experiment_data/`: generated outputs, evaluated results, summaries, and formal-release snapshots

## Project Summary

Q-AssertBench is a benchmark framework for studying assertion generation in quantum programs. The repository provides the task set, generation clients, execution-based evaluator, and reporting scripts needed to reproduce or extend the released experiments, with particular emphasis on the evaluation stage that turns raw generations into benchmark outcomes.
