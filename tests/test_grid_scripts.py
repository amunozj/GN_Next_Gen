"""The long-run scripts: the properties an unattended overnight run depends on.

These are cheap static checks, not end-to-end runs. They exist because the
failure they guard against is expensive and silent: a grid that starts, fails
every cell for a reason unrelated to the science, and reports each one as an
ordinary cell failure.
"""

from __future__ import annotations

import ast
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

from gn_next_gen.paths import REPO_ROOT

SCRIPTS = REPO_ROOT / "scripts"
ORCHESTRATOR = SCRIPTS / "run_step2_grid.sh"


def _load(name: str):
    """Import a script by path, without running its main()."""
    import importlib.util

    key = f"_script_{name}"
    spec = importlib.util.spec_from_file_location(key, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    # Registered before exec: pickle resolves a function by module name, so an
    # unregistered module makes every function in it look unpicklable and the
    # test would fail on its own fixture rather than on the code.
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("script", "worker"),
    [("run_baselines", "run_cell"), ("generate_benchmark", "build_one")],
)
def test_the_parallel_worker_is_picklable(script, worker):
    """``ProcessPoolExecutor`` pickles the callable it is handed.

    A nested function cannot be pickled, so submitting a closure makes every
    task fail with "Can't get local object" -- which is what ``--jobs 8`` did
    on a 189-cell trial run, reporting each one as a cell failure rather than
    as the one structural fault it was.
    """
    module = _load(script)
    function = getattr(module, worker)
    assert function.__qualname__ == worker, (
        f"{worker} is nested inside {function.__qualname__}; a worker must be module-level"
    )
    pickle.loads(pickle.dumps(function))


@pytest.mark.parametrize("script", ["run_baselines", "generate_benchmark"])
def test_nothing_local_is_submitted_to_a_process_pool(script):
    """The same fault, caught at the call site rather than at the symbol."""
    tree = ast.parse((SCRIPTS / f"{script}.py").read_text())
    local_functions = {
        node.name
        for parent in ast.walk(tree)
        if isinstance(parent, ast.FunctionDef)
        for node in parent.body
        if isinstance(node, ast.FunctionDef)
    }
    submitted = [
        node.args[0].id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "submit"
        and node.args
        and isinstance(node.args[0], ast.Name)
    ]
    offenders = sorted(set(submitted) & local_functions)
    assert offenders == [], f"{script} submits locally-defined {offenders} to a pool"


def test_every_long_script_writes_its_own_lineage_file():
    """Several share one log directory, and ADR-002 wants lineage per run."""
    names = set()
    for script in ("generate_benchmark", "run_baselines", "score_benchmark"):
        text = (SCRIPTS / f"{script}.py").read_text()
        assert "write_lineage(" in text, script
        assert "filename=" in text, f"{script} does not name its lineage file"
        names.add(script)
    assert len(names) == 3


def test_the_orchestrator_is_executable_and_parses():
    assert ORCHESTRATOR.is_file()
    assert ORCHESTRATOR.stat().st_mode & 0o111, "run_step2_grid.sh is not executable"
    result = subprocess.run(
        ["bash", "-n", str(ORCHESTRATOR)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_the_orchestrator_respects_the_one_hard_barrier():
    """Generation must complete before anything that reads a world starts.

    The runners glob for *complete* realizations, so one started early would
    quietly process a subset and look finished.
    """
    text = ORCHESTRATOR.read_text()
    generate_at = text.index("stage generate")
    for later in ("stage baselines", "stage structural", "stage sampling", "stage figures"):
        assert text.index(later) > generate_at, f"{later} is ordered before generation"
    # And the generation stage aborts the run rather than letting them proceed.
    assert "ABORTING" in text[generate_at : generate_at + 1200]


def test_the_orchestrator_reports_help_without_running_anything():
    result = subprocess.run(
        [sys.executable and "bash", str(ORCHESTRATOR), "--help"],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0
    assert "--detach" in result.stdout
    assert "--status" in result.stdout
    assert "START" not in result.stdout, "--help started a stage"


def test_status_is_safe_before_anything_has_run(tmp_path):
    """--status must never be the thing that launches the grid."""
    text = ORCHESTRATOR.read_text()
    status_case = text.index("--status) status; exit 0 ;;")
    first_stage = text.index("stage a157")
    assert status_case < first_stage
    assert Path(ORCHESTRATOR).read_text().count("exit 0 ;;") >= 1


def _dry_run(*args: str) -> str:
    result = subprocess.run(
        ["bash", str(ORCHESTRATOR), "--dry-run", *args],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_a_dry_run_lists_every_stage_in_order():
    out = _dry_run()
    order = ["a157", "generate", "baselines", "structural", "sampling", "score", "figures"]
    positions = [out.index(f"would run  {name}") for name in order]
    assert positions == sorted(positions), out


def test_repeated_name_flags_are_emitted_once_for_argparse():
    """``nargs="*"`` keeps only the LAST occurrence of a repeated flag.

    Passing ``--scenarios S1 --scenarios S2 --scenarios S5`` therefore runs one
    scenario, silently. The orchestrator must emit ``--scenarios S1 S2 S5``.
    """
    out = _dry_run("--scenarios", "S1", "S2", "S5", "--methods", "HoSc98", "CEA17")
    generate = next(line for line in out.splitlines() if "would run  generate" in line)
    assert "--scenarios S1 S2 S5" in generate
    assert generate.count("--scenarios") == 1

    baselines = next(line for line in out.splitlines() if "would run  baselines" in line)
    assert "--methods HoSc98 CEA17" in baselines
    assert baselines.count("--methods") == 1


def test_the_jobs_flag_reaches_every_runner():
    out = _dry_run("--jobs", "12")
    for name in ("generate", "baselines", "structural"):
        line = next(line for line in out.splitlines() if f"would run  {name}" in line)
        assert "--jobs 12" in line, line


def test_a_dry_run_writes_nothing():
    before = (
        sorted((REPO_ROOT / "outputs" / "step2" / "logs").glob("*"))
        if (REPO_ROOT / "outputs" / "step2" / "logs").is_dir()
        else []
    )
    _dry_run()
    after = (
        sorted((REPO_ROOT / "outputs" / "step2" / "logs").glob("*"))
        if (REPO_ROOT / "outputs" / "step2" / "logs").is_dir()
        else []
    )
    assert before == after
