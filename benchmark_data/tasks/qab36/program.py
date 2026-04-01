from __future__ import annotations

from qiskit import QuantumCircuit, transpile

from qasserbench.execution.backends import get_backend
from qasserbench.execution.interfaces import (
    ExecutionConfig,
    ExecutionResult,
    GoldAssertionResult,
    ProgramDefinition,
)
from qasserbench.execution.runtime import normalize_counts


REGISTER_WIDTH = 3
XOR_MASK = "101"
TOTAL_QUBITS = REGISTER_WIDTH * 2


def build_circuit() -> QuantumCircuit:
    circuit = QuantumCircuit(TOTAL_QUBITS, TOTAL_QUBITS)
    circuit.h(0)
    circuit.h(1)
    circuit.h(2)
    circuit.cx(0, 3)
    circuit.cx(1, 4)
    circuit.cx(2, 5)
    circuit.x(3)
    circuit.x(5)
    circuit.measure(range(TOTAL_QUBITS), range(TOTAL_QUBITS))
    return circuit


def build_program() -> QuantumCircuit:
    return build_circuit()


def run_program(config: ExecutionConfig) -> ExecutionResult:
    backend = get_backend(config.backend)
    transpiled = transpile(build_circuit(), backend)
    run_kwargs = {"shots": config.shots}
    if config.seed is not None:
        run_kwargs["seed_simulator"] = config.seed
    job = backend.run(transpiled, **run_kwargs)
    counts = normalize_counts(job.result().get_counts())
    return ExecutionResult(
        counts=counts,
        shots=config.shots,
        backend=config.backend,
        metadata=dict(config.metadata),
    )


def _placeholder_gold_assertion(_: ExecutionResult) -> GoldAssertionResult:
    return GoldAssertionResult(
        passed=False,
        details={"note": "Loader should replace this placeholder gold evaluator."},
    )


PROGRAM = ProgramDefinition(
    task_id="QAB36",
    build_program=build_program,
    run_program=run_program,
    evaluate_gold_assertion=_placeholder_gold_assertion,
)
