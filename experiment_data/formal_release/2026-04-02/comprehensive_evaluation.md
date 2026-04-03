# Comprehensive Evaluation

Date: 2026-04-02

This report summarizes the original six complete formal-generation runs that were included when this comparison was first produced:

- `deepseek/deepseek-v3.2`
- `qwen/qwen3.5-397b-a17b`
- `gpt-5.4`
- `claude-sonnet-4-20250514`
- `gemini-3.1-flash-lite-preview`
- `meta-llama/llama-4-scout`

The later-added `gemini-2.5-flash` and `gemini-3-flash-preview` formal runs are included in the release snapshot and each has its own `trial_results.jsonl` and `summary.json`, but they are not folded into the six-model comparison tables below.

All model summaries below are based on `37` tasks with `20` trials per task (`740` trials/model).

## Model-Level Results

| Model | Pass Rate | Pass@1 | Pass@5 | Mean Alignment | Invalid |
| --- | ---: | ---: | ---: | ---: | ---: |
| `gpt-5.4` | `0.7892` | `0.9730` | `0.8919` | `0.6641` | `1` |
| `claude-sonnet-4` | `0.7149` | `0.9459` | `0.7838` | `0.6386` | `1` |
| `qwen3.5-397b` | `0.6784` | `0.9459` | `0.8649` | `0.5762` | `1` |
| `deepseek-v3.2` | `0.5730` | `1.0000` | `0.8378` | `0.5572` | `59` |
| `gemini-3.1-flash-lite` | `0.5473` | `0.8108` | `0.6486` | `0.5688` | `4` |
| `llama-4-scout` | `0.3230` | `0.7568` | `0.4324` | `0.3810` | `249` |

## Difficulty Subsets

### High-Difficulty Reasoning

Tasks: `QAB21`, `QAB26`, `QAB34`

| Model | Subset Pass Rate |
| --- | ---: |
| `gpt-5.4` | `0.6667` |
| `deepseek-v3.2` | `0.2333` |
| `qwen3.5-397b` | `0.2167` |
| `llama-4-scout` | `0.1833` |
| `claude-sonnet-4` | `0.1000` |
| `gemini-3.1-flash-lite` | `0.0333` |
| **Average** | **`0.2389`** |

This is the hardest curated subset in the benchmark.

### Long-Source

Tasks: `QAB32`, `QAB33`, `QAB34`

| Model | Subset Pass Rate |
| --- | ---: |
| `gpt-5.4` | `0.7333` |
| `deepseek-v3.2` | `0.5833` |
| `qwen3.5-397b` | `0.5833` |
| `claude-sonnet-4` | `0.5167` |
| `gemini-3.1-flash-lite` | `0.3333` |
| `llama-4-scout` | `0.3333` |
| **Average** | **`0.5139`** |

This subset is moderately difficult overall, but its hardness is carried mainly by `QAB32` and `QAB34`.

### Structured Trap

Tasks: `QAB35`, `QAB36`, `QAB37`

| Model | Subset Pass Rate |
| --- | ---: |
| `qwen3.5-397b` | `0.9000` |
| `gpt-5.4` | `0.8667` |
| `claude-sonnet-4` | `0.8500` |
| `gemini-3.1-flash-lite` | `0.6333` |
| `deepseek-v3.2` | `0.5167` |
| `llama-4-scout` | `0.0500` |
| **Average** | **`0.6361`** |

This subset is not uniformly hard across models; only `QAB37` behaves like a consistently difficult trap task.

## Cross-Model Hardest Tasks

Sorted by mean pass rate across all six models:

| Task | Mean Pass Rate | Min Model | Max Model |
| --- | ---: | ---: | ---: |
| `QAB21` | `0.1250` | `0.0000` | `0.4500` |
| `QAB29` | `0.1250` | `0.0000` | `0.4000` |
| `QAB18` | `0.1667` | `0.0000` | `0.5500` |
| `QAB30` | `0.2083` | `0.0500` | `0.4000` |
| `QAB34` | `0.2250` | `0.0000` | `0.9500` |
| `QAB06` | `0.2667` | `0.0000` | `1.0000` |
| `QAB15` | `0.3167` | `0.0000` | `0.7000` |
| `QAB20` | `0.3250` | `0.0000` | `0.8000` |
| `QAB26` | `0.3667` | `0.1000` | `0.9500` |
| `QAB05` | `0.4250` | `0.0000` | `0.9500` |

## Cross-Model Easiest Tasks

| Task | Mean Pass Rate | Min Model | Max Model |
| --- | ---: | ---: | ---: |
| `QAB08` | `0.8417` | `0.2500` | `1.0000` |
| `QAB09` | `0.8417` | `0.6500` | `1.0000` |
| `QAB02` | `0.8500` | `0.4500` | `1.0000` |
| `QAB12` | `0.8583` | `0.5000` | `1.0000` |
| `QAB19` | `0.8833` | `0.6500` | `1.0000` |
| `QAB16` | `0.8917` | `0.7000` | `1.0000` |
| `QAB25` | `0.8917` | `0.6000` | `1.0000` |
| `QAB14` | `0.9000` | `0.7000` | `1.0000` |
| `QAB22` | `0.9417` | `0.8000` | `1.0000` |
| `QAB10` | `0.9833` | `0.9500` | `1.0000` |

## Hardest Categories

Sorted by average pass rate across all six models:

| Category | Avg Pass Rate | Min Model | Max Model |
| --- | ---: | ---: | ---: |
| `qft_periodicity` | `0.1250` | `0.0000` | `0.4250` |
| `qaoa_solution_pattern` | `0.2250` | `0.0000` | `0.9500` |
| `bernstein_vazirani_distribution` | `0.3167` | `0.0000` | `0.7000` |
| `shor_periodicity` | `0.4333` | `0.2000` | `0.7000` |
| `grover_distribution` | `0.4417` | `0.0000` | `0.9000` |
| `algorithmic_semantics_oracle_relation` | `0.4792` | `0.0500` | `0.8750` |

## Notes

- Evaluation was rerun after fixing a runner bug where candidate snippets could mutate the shared `counts` dictionary and contaminate later gold-alignment checks.
- The updated runner now gives each candidate a private copy of `counts`, so these results should be treated as the canonical comparison for this release snapshot.
