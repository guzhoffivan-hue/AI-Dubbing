"""The core's stage table must not drift from scripts/cli.py.

The engine's own driver declares the same twelve stages a second time, and that
duplication already caused one real defect: cli.py declared the wrong artifact for
``segment_merge``, so a stale ``translated_segments.json`` was reused and the dub
went silent at 3:30. Until the table has a single source, this test makes the two
copies fail loudly instead of drifting quietly.

It reads cli.py as source (no import, no side effects) and compares the declared
phase order, the script of each stage and the artifact each stage publishes.
"""

import ast
from pathlib import Path

from app.adapters import ENGINE_STAGES

CLI = Path(__file__).resolve().parents[2] / "scripts" / "cli.py"


def cli_tree() -> ast.Module:
    return ast.parse(CLI.read_text(encoding="utf-8"))


def find_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node

    raise AssertionError(f"{name}() not found in {CLI}")


def phase_tuples(process: ast.FunctionDef) -> list[ast.Tuple]:
    """The tuples that build ``phases``, in source order.

    ``ast.walk`` is breadth first, which would reorder the stages, so the tuples are
    collected and then sorted by line: the declaration order is part of the contract.
    """

    found: list[ast.Tuple] = []

    for node in ast.walk(process):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "phases"
            for target in node.targets
        ):
            if isinstance(node.value, ast.List):
                for element in node.value.elts:
                    if isinstance(element, ast.Tuple):
                        found.append(element)

        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("append", "extend")
        ):
            for argument in node.args:
                if isinstance(argument, ast.Tuple):
                    found.append(argument)
                elif isinstance(argument, ast.List):
                    for element in argument.elts:
                        if isinstance(element, ast.Tuple):
                            found.append(element)

    return sorted(found, key=lambda phase: phase.lineno)


def script_of(phase: ast.Tuple) -> str | None:
    for node in ast.walk(phase):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "run_script" and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    return first.value

    return None


def declared_phases(process: ast.FunctionDef) -> tuple[list[str], dict[str, str]]:
    names: list[str] = []
    scripts: dict[str, str] = {}

    for phase in phase_tuples(process):
        if not phase.elts:
            continue

        first = phase.elts[0]
        if not (isinstance(first, ast.Constant) and isinstance(first.value, str)):
            continue

        name = first.value

        if name not in names:
            names.append(name)

        script = script_of(phase)

        if script is not None:
            scripts.setdefault(name, script)

    return names, scripts


def declared_outputs(process: ast.FunctionDef) -> dict[str, str | None]:
    for node in ast.walk(process):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "outputs"
            for target in node.targets
        ):
            if isinstance(node.value, ast.Dict):
                outputs: dict[str, str | None] = {}

                for key, value in zip(node.value.keys, node.value.values):
                    if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
                        continue

                    outputs[key.value] = (
                        value.value
                        if isinstance(value, ast.Constant) and isinstance(value.value, str)
                        else None
                    )

                return outputs

    raise AssertionError(f"no outputs table found in process() of {CLI}")


def test_cli_and_core_declare_the_same_stages_in_the_same_order():
    process = find_function(cli_tree(), "process")
    names, _scripts = declared_phases(process)

    assert names == [spec.name for spec in ENGINE_STAGES]


def test_each_stage_runs_the_script_the_core_expects():
    process = find_function(cli_tree(), "process")
    _names, scripts = declared_phases(process)

    assert scripts == {spec.name: spec.script for spec in ENGINE_STAGES}


def test_each_stage_publishes_the_artifact_the_core_expects():
    process = find_function(cli_tree(), "process")
    outputs = declared_outputs(process)

    assert outputs == {spec.name: spec.artifact for spec in ENGINE_STAGES}
