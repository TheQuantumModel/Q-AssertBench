"""Prompt rendering helpers for assertion-generation tasks."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from qasserbench.benchmark.loader import LoadedTaskAssets


PROMPT_TEMPLATE_VERSION = "assertion_snippet_v2"
COMMON_PROMPT_INTRO = (
    "Write a short Python assertion snippet for this quantum-program benchmark task."
)


@dataclass(frozen=True)
class PromptRenderingContext:
    """Inspectable prompt-rendering summary for one task."""

    prompt_version: str
    program_path: Path
    observation_stage: str | None
    probe_function_names: tuple[str, ...]
    selected_constant_names: tuple[str, ...]
    selected_function_names: tuple[str, ...]
    source_excerpt: str
    task_specification: str


def _program_module_path(assets: "LoadedTaskAssets") -> Path:
    relative_path, _ = assets.task.program_entry.split(":", 1)
    return (assets.task.task_dir / relative_path).resolve()


def _read_program_source(assets: "LoadedTaskAssets") -> str:
    return _program_module_path(assets).read_text(encoding="utf-8")


def _task_specification_text(prompt_text: str) -> str:
    kept_lines: list[str] = []
    for line in prompt_text.splitlines():
        stripped = line.strip()
        if not stripped:
            if kept_lines and kept_lines[-1]:
                kept_lines.append("")
            continue
        if stripped.lower().startswith("output "):
            continue
        kept_lines.append(stripped)
    return "\n".join(kept_lines).strip()


def _is_uppercase_assignment(node: ast.AST) -> bool:
    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, ast.AnnAssign):
        targets = [node.target]
    else:
        return False

    for target in targets:
        if isinstance(target, ast.Name) and target.id.isupper():
            return True
    return False


def _function_defs(module: ast.Module) -> dict[str, ast.FunctionDef]:
    return {
        node.name: node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
    }


def _called_local_function_names(node: ast.AST, local_names: set[str]) -> set[str]:
    called: set[str] = set()
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        if isinstance(child.func, ast.Name) and child.func.id in local_names:
            called.add(child.func.id)
    return called


def _selected_function_names(module: ast.Module) -> list[str]:
    defs = _function_defs(module)
    local_names = set(defs)
    selected: set[str] = {name for name in ("build_circuit", "run_program") if name in defs}

    queue = list(selected)
    while queue:
        current = queue.pop()
        for called_name in _called_local_function_names(defs[current], local_names):
            if called_name not in selected:
                selected.add(called_name)
                queue.append(called_name)

    return sorted(selected, key=lambda name: defs[name].lineno)


def _selected_constant_names(
    module: ast.Module,
    selected_functions: Iterable[str],
) -> list[str]:
    defs = _function_defs(module)
    referenced: set[str] = set()
    for function_name in selected_functions:
        function_node = defs[function_name]
        for child in ast.walk(function_node):
            if isinstance(child, ast.Name) and child.id.isupper():
                referenced.add(child.id)

    constants: list[tuple[int, str]] = []
    for node in module.body:
        if not _is_uppercase_assignment(node):
            continue
        target_names: list[str] = []
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    target_names.append(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target_names.append(node.target.id)

        for name in target_names:
            if name in referenced:
                constants.append((node.lineno, name))

    return [name for _, name in sorted(constants)]


def _node_source_segments(
    source_text: str,
    module: ast.Module,
    *,
    selected_constant_names: Iterable[str],
    selected_function_names: Iterable[str],
) -> list[str]:
    defs = _function_defs(module)
    selected_constants = set(selected_constant_names)
    segments: list[tuple[int, str]] = []

    for node in module.body:
        segment = ast.get_source_segment(source_text, node)
        if not segment:
            continue

        if isinstance(node, ast.FunctionDef) and node.name in selected_function_names:
            segments.append((node.lineno, segment.strip()))
            continue

        if not _is_uppercase_assignment(node):
            continue

        names: list[str] = []
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names = [node.target.id]
        if any(name in selected_constants for name in names):
            segments.append((node.lineno, segment.strip()))

    return [segment for _, segment in sorted(segments)]


def _run_program_node(module: ast.Module) -> ast.FunctionDef | None:
    return _function_defs(module).get("run_program")


def _probe_function_names(module: ast.Module) -> list[str]:
    run_program = _run_program_node(module)
    if run_program is None:
        return []
    local_names = set(_function_defs(module))
    probe_names = [
        name
        for name in _called_local_function_names(run_program, local_names)
        if "probe" in name
    ]
    return sorted(probe_names)


def _subscript_key(node: ast.Subscript) -> str | None:
    slice_node = node.slice
    if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, str):
        return slice_node.value
    if hasattr(ast, "Index") and isinstance(slice_node, ast.Index):
        value = slice_node.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return value.value
    return None


def _observation_stage(module: ast.Module) -> str | None:
    run_program = _run_program_node(module)
    if run_program is None:
        return None

    for node in ast.walk(run_program):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Subscript):
                continue
            if not isinstance(target.value, ast.Name) or target.value.id != "metadata":
                continue
            if _subscript_key(target) != "observation_stage":
                continue
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                return node.value.value
    return None


def _render_observation_contract(module: ast.Module) -> str:
    stage = _observation_stage(module)
    probe_names = _probe_function_names(module)

    lines = [
        "Execution context:",
        "- Your snippet is executed after `run_program(...)` has produced `counts`.",
        "- The snippet itself runs with `counts`, `shots`, and `metadata` already available.",
    ]

    if probe_names:
        rendered_probe_names = ", ".join(f"`{name}()`" for name in probe_names)
        lines.append(
            f"- The provided `counts` are produced by the probe function {rendered_probe_names} rather than the full task pipeline."
        )
    else:
        lines.append("- The provided `counts` come from the task's standard `run_program(...)` execution path.")

    if stage:
        lines.append(f"- Observation stage: `{stage}`.")

    return "\n".join(lines)


def _render_source_excerpt(assets: "LoadedTaskAssets") -> str:
    source_text = _read_program_source(assets)
    module = ast.parse(source_text)
    selected_function_names = _selected_function_names(module)
    selected_constant_names = _selected_constant_names(module, selected_function_names)
    segments = _node_source_segments(
        source_text,
        module,
        selected_constant_names=selected_constant_names,
        selected_function_names=selected_function_names,
    )

    if not segments:
        return source_text.strip()

    return "\n\n".join(segments)


def inspect_task_prompt(assets: "LoadedTaskAssets") -> PromptRenderingContext:
    """Build inspectable prompt-rendering context for one task."""
    source_text = _read_program_source(assets)
    module = ast.parse(source_text)
    selected_function_names = tuple(_selected_function_names(module))
    selected_constant_names = tuple(_selected_constant_names(module, selected_function_names))
    task_spec = _task_specification_text(assets.prompt_text)
    source_excerpt = _render_source_excerpt(assets)

    return PromptRenderingContext(
        prompt_version=PROMPT_TEMPLATE_VERSION,
        program_path=_program_module_path(assets),
        observation_stage=_observation_stage(module),
        probe_function_names=tuple(_probe_function_names(module)),
        selected_constant_names=selected_constant_names,
        selected_function_names=selected_function_names,
        source_excerpt=source_excerpt,
        task_specification=task_spec,
    )


def render_task_prompt(assets: "LoadedTaskAssets") -> str:
    """Render the final model-facing prompt for one benchmark task."""

    context = inspect_task_prompt(assets)
    module = ast.parse(_read_program_source(assets))
    observation_contract = _render_observation_contract(module)

    sections = [
        COMMON_PROMPT_INTRO,
        f"Prompt template version: `{context.prompt_version}`.",
        "\n".join(
            [
                "Return one Python code block containing only the assertion snippet.",
                "Keep the snippet self-contained, using `assert` statements and simple local variables.",
                "If you need constants from the source code, write their values directly in the snippet or read them from `metadata` when available.",
                "Avoid imports, helper definitions, and extra explanation outside the code block.",
            ]
        ),
        "Task specification:\n" + context.task_specification,
        observation_contract,
        "Relevant source excerpt from `program.py`:\n```python\n"
        + context.source_excerpt
        + "\n```",
    ]
    return "\n\n".join(section.strip() for section in sections if section.strip())
