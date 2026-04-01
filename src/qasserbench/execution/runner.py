"""Helpers for executing generated assertion candidates."""

from __future__ import annotations

from types import MappingProxyType

from qasserbench.execution.interfaces import (
    AssertionCheckResult,
    ExecutionConfig,
    FaultCheckResult,
    ProgramDefinition,
    TrialExecutionRecord,
)
from qasserbench.generation.artifacts import CandidateAssertionArtifact


def _evaluate_candidate_against_counts(
    artifact: CandidateAssertionArtifact,
    counts: dict[str, int],
    *,
    shots: int,
    backend: str,
    metadata: dict[str, object],
) -> AssertionCheckResult:
    if not artifact.is_usable:
        return AssertionCheckResult(
            passed=False,
            error_type="unusable_candidate",
            message="Candidate artifact is not usable.",
        )

    try:
        compiled = compile(artifact.code or "", "<candidate-assertion>", "exec")
        # Use a shared scope for globals/locals so comprehensions and generator
        # expressions can still resolve names assigned earlier in the snippet.
        execution_scope = {
            "counts": counts,
            "shots": shots,
            "backend": backend,
            "metadata": MappingProxyType(dict(metadata)),
        }
        exec(compiled, execution_scope, execution_scope)
    except AssertionError as exc:
        return AssertionCheckResult(
            passed=False,
            error_type="assertion_failed",
            message=str(exc) or None,
        )
    except Exception as exc:  # noqa: BLE001
        return AssertionCheckResult(
            passed=False,
            error_type="runtime_error",
            message=str(exc) or exc.__class__.__name__,
        )

    return AssertionCheckResult(passed=True)


def run_candidate_trial(
    *,
    program: ProgramDefinition,
    fault_programs: dict[str, ProgramDefinition],
    artifact: CandidateAssertionArtifact,
    config: ExecutionConfig,
) -> TrialExecutionRecord:
    nominal_execution = program.run_program(config)
    nominal_assertion = _evaluate_candidate_against_counts(
        artifact,
        nominal_execution.counts,
        shots=nominal_execution.shots,
        backend=nominal_execution.backend,
        metadata=nominal_execution.metadata,
    )

    fault_results: list[FaultCheckResult] = []
    for fault_id, fault_program in fault_programs.items():
        fault_execution = fault_program.run_program(config)
        fault_assertion = _evaluate_candidate_against_counts(
            artifact,
            fault_execution.counts,
            shots=fault_execution.shots,
            backend=fault_execution.backend,
            metadata=fault_execution.metadata,
        )
        fault_results.append(
            FaultCheckResult(
                fault_id=fault_id,
                execution_result=fault_execution,
                assertion_result=fault_assertion,
            )
        )

    return TrialExecutionRecord(
        nominal_execution=nominal_execution,
        nominal_assertion=nominal_assertion,
        fault_results=fault_results,
    )
