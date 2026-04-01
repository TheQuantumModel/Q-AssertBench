"""Evaluate stored generation records into trial-level benchmark results."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from qasserbench.benchmark.loader import load_task_assets
from qasserbench.evaluation.alignment import compare_candidate_to_gold
from qasserbench.evaluation.classify import classify_trial
from qasserbench.execution.backends import DEFAULT_SIMULATOR_BACKEND
from qasserbench.execution.interfaces import ExecutionConfig
from qasserbench.execution.runner import run_candidate_trial
from qasserbench.generation.driver import read_generation_records
from qasserbench.generation.extract import extract_candidate_assertion
from qasserbench.reporting.io import write_trial_results


def _build_task_category(family: str, property_type: str) -> str:
    return f"{family}_{property_type}"


def _generation_status(candidate_diagnostics: list[str], *, is_usable: bool) -> str:
    if is_usable:
        return "success"
    if "empty_response" in candidate_diagnostics:
        return "empty_response"
    return "format_error"


def _nominal_status(*, is_usable: bool, nominal_passed: bool, nominal_error: str | None) -> str:
    if not is_usable:
        return "unusable_candidate"
    if nominal_passed:
        return "pass"
    return nominal_error or "assertion_failed"


def _fault_status(
    *,
    is_usable: bool,
    fault_results: list[dict[str, str | bool | None]],
) -> str:
    if not fault_results:
        return "not_applicable"
    if not is_usable:
        return "unusable_candidate"

    if any(result["assertion_error"] == "runtime_error" for result in fault_results):
        return "runtime_error"

    detected = sum(
        1
        for result in fault_results
        if result["assertion_passed"] is False and result["assertion_error"] == "assertion_failed"
    )
    if detected == len(fault_results):
        return "detected"
    if detected > 0:
        return "partially_detected"
    return "missed"


def evaluate_generation_records(
    *,
    input_path: str | Path,
    output_path: str | Path,
    backend: str = DEFAULT_SIMULATOR_BACKEND,
    seed: int | None = 7,
) -> Path:
    """Convert raw generation outputs into evaluated trial records."""

    generation_records = read_generation_records(input_path)
    evaluated_records: list[dict[str, Any]] = []

    for generation_record in generation_records:
        manifest_path = generation_record["manifest_path"]
        assets = load_task_assets(manifest_path)
        artifact = extract_candidate_assertion(
            raw_response=str(generation_record["raw_response"]),
            extraction_mode=assets.task.insertion_mode,
        )
        config = ExecutionConfig(
            shots=assets.task.shots,
            backend=backend,
            seed=seed,
            metadata={
                "model_id": generation_record["model_id"],
                "trial_index": generation_record["trial_index"],
            },
        )
        trial = run_candidate_trial(
            program=assets.program,
            fault_programs=assets.fault_programs,
            artifact=artifact,
            config=config,
        )
        alignment = compare_candidate_to_gold(
            artifact,
            assets.task.gold_metadata,
            assets.gold_source,
            trial=trial,
            gold_evaluator=assets.program.evaluate_gold_assertion,
        )
        classification = classify_trial(artifact, trial, alignment)
        gold_nominal = assets.program.evaluate_gold_assertion(trial.nominal_execution)
        fault_results = [
            {
                "fault_id": fault_result.fault_id,
                "assertion_passed": fault_result.assertion_result.passed,
                "assertion_error": fault_result.assertion_result.error_type,
            }
            for fault_result in trial.fault_results
        ]
        generation_status = _generation_status(
            list(artifact.diagnostics),
            is_usable=artifact.is_usable,
        )
        nominal_status = _nominal_status(
            is_usable=artifact.is_usable,
            nominal_passed=trial.nominal_assertion.passed,
            nominal_error=trial.nominal_assertion.error_type,
        )
        fault_status = _fault_status(
            is_usable=artifact.is_usable,
            fault_results=fault_results,
        )

        evaluated_records.append(
            {
                "model_id": generation_record["model_id"],
                "provider_model_id": generation_record.get("provider_model_id"),
                "task_id": assets.task.task_id,
                "task_category": _build_task_category(
                    assets.task.family,
                    assets.task.property_type,
                ),
                "trial_index": generation_record["trial_index"],
                "manifest_path": manifest_path,
                "raw_response": generation_record["raw_response"],
                "raw_payload": generation_record.get("raw_payload", {}),
                "generation_temperature": generation_record.get("generation_temperature"),
                "generation_max_output_tokens": generation_record.get("generation_max_output_tokens"),
                "prompt_tokens": generation_record.get("prompt_tokens"),
                "completion_tokens": generation_record.get("completion_tokens"),
                "total_tokens": generation_record.get("total_tokens"),
                "generation_status": generation_status,
                "candidate_code": artifact.code,
                "candidate_diagnostics": list(artifact.diagnostics),
                "nominal_status": nominal_status,
                "fault_status": fault_status,
                "overall_outcome": classification.outcome,
                "outcome": classification.outcome,
                "alignment_label": classification.alignment_label,
                "alignment_score": alignment.score,
                "alignment_components": dict(alignment.components),
                "alignment_notes": list(alignment.notes),
                "failure_tags": list(classification.failure_tags),
                "fault_detection_rate": classification.fault_detection_rate,
                "gold_nominal_passed": gold_nominal.passed,
                "gold_nominal_details": dict(gold_nominal.details),
                "nominal_assertion_passed": trial.nominal_assertion.passed,
                "nominal_assertion_error": trial.nominal_assertion.error_type,
                "fault_results": fault_results,
            }
        )

    return write_trial_results(evaluated_records, output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate raw generation outputs into trial results.")
    parser.add_argument("input", type=Path, help="Path to generation_records.jsonl")
    parser.add_argument("output", type=Path, help="Path to output trial_results.jsonl")
    parser.add_argument(
        "--backend",
        default=DEFAULT_SIMULATOR_BACKEND,
        help="Backend used for program execution",
    )
    parser.add_argument("--seed", type=int, default=7, help="Optional simulator seed")
    args = parser.parse_args()

    output_path = evaluate_generation_records(
        input_path=args.input,
        output_path=args.output,
        backend=args.backend,
        seed=args.seed,
    )
    print(f"wrote trial results to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
