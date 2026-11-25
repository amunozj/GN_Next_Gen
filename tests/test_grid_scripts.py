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


def test_a_dry_run_lists_every_stage_and_respects_the_real_barriers():
    """Every stage appears, and the orderings the script actually promises hold.

    This asserted a strict source order of all seven and was **flaky**: three of
    the stages are launched as a concurrent group, so their lines interleave
    nondeterministically.  It passed on an idle machine and failed twice in three
    runs at a load average of 147, which is exactly the sort of failure that gets
    dismissed as noise.

    What the script does promise is the barrier structure, and that is what is
    pinned here: nothing that reads a world may start before `generate`, scoring
    comes after the fits it scores, and figures come last.
    """
    out = _dry_run()
    stages = ["a157", "generate", "baselines", "structural", "sampling", "score", "figures"]
    at = {}
    for name in stages:
        token = f"would run  {name}"
        assert token in out, f"{name} missing from the dry run:\n{out}"
        at[name] = out.index(token)

    # The one hard dependency: generation is a barrier for everything that reads a
    # world.  A runner started early globs for *complete* realizations and would
    # quietly process a subset while looking finished.
    for reader in ("baselines", "structural", "sampling"):
        assert at["generate"] < at[reader], f"{reader} must come after generate:\n{out}"
    assert at["baselines"] < at["score"], out
    assert at["score"] < at["figures"], out
    assert at["figures"] == max(at.values()), out


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


def test_progress_records_the_orchestrators_run_id(tmp_path, monkeypatch):
    """--status tells this run's files from an earlier run's by run_id.

    An mtime gap cannot do it: an early stage legitimately finishes long before
    a late one, and treating that as staleness flagged a healthy 459-cell grid
    as leftovers.
    """
    from gn_next_gen.benchmark.progress import Progress

    monkeypatch.setenv("GN_RUN_ID", "20261122T173352-42")
    first = Progress(tmp_path / "a_progress.json", total=2, label="a")
    second = Progress(tmp_path / "b_progress.json", total=2, label="b")
    import json as _json

    assert _json.loads(first.path.read_text())["run_id"] == "20261122T173352-42"
    assert _json.loads(second.path.read_text())["run_id"] == "20261122T173352-42"


def test_progress_falls_back_to_its_pid_when_run_standalone(tmp_path, monkeypatch):
    import json as _json
    import os

    from gn_next_gen.benchmark.progress import Progress

    monkeypatch.delenv("GN_RUN_ID", raising=False)
    progress = Progress(tmp_path / "c_progress.json", total=1, label="c")
    assert _json.loads(progress.path.read_text())["run_id"] == str(os.getpid())


def test_the_orchestrator_exports_a_run_id_before_any_stage():
    text = ORCHESTRATOR.read_text()
    assert "export GN_RUN_ID=" in text
    assert text.index("export GN_RUN_ID=") < text.index("stage a157")


# --- staleness: a resumed run must not reuse work the current code would not write


def test_a_world_from_an_older_generator_is_not_complete(tmp_path):
    """The guard that makes the re-run safe to launch without --force.

    `is_complete` used to ask only whether the files were there, so changing the
    generator left every stage skipping worlds it would no longer produce, and the
    scores came out of a mixture of two generators with nothing to say which.
    """
    import json

    from gn_next_gen.benchmark.store import (
        GENERATOR_VERSION,
        OPEN_FILES,
        SEALED_FILES,
        is_complete,
        realization_dir,
    )

    base = realization_dir("S1", 7, root=tmp_path)
    (base / "open").mkdir(parents=True)
    (base / "sealed").mkdir(parents=True)
    for name in OPEN_FILES:
        (base / "open" / name).write_bytes(b"")
    for name in SEALED_FILES:
        (base / "sealed" / name).write_bytes(b"")

    (base / "config.json").write_text(json.dumps({"generator_version": GENERATOR_VERSION - 1}))
    assert not is_complete("S1", 7, root=tmp_path)

    (base / "config.json").write_text(json.dumps({"generator_version": GENERATOR_VERSION}))
    assert is_complete("S1", 7, root=tmp_path)

    (base / "config.json").write_text("{not json")
    assert not is_complete("S1", 7, root=tmp_path)


def test_a_fit_without_the_uncertainty_columns_is_rerun(tmp_path):
    """Same argument one layer up: the files exist, the information does not."""
    import importlib.util

    import pandas as pd

    spec = importlib.util.spec_from_file_location(
        "run_baselines", Path(__file__).resolve().parents[1] / "scripts" / "run_baselines.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cell = module.cell_dir("S1", 7, root=tmp_path)
    cell.mkdir(parents=True)
    old = pd.DataFrame({"date_ord": [1], "gn": [1.0], "gn_sd": [0.5]})
    old.to_parquet(cell / "HoSc98_series.parquet", index=False)
    assert not module.cell_done("S1", 7, "HoSc98", root=tmp_path)

    current = old.assign(gn_sd_factor=0.4, gn_sd_spread=0.3)
    current.to_parquet(cell / "HoSc98_series.parquet", index=False)
    assert module.cell_done("S1", 7, "HoSc98", root=tmp_path)


# --- Step 3's runner: staleness, not just presence ------------------------


def _load_runner():
    import importlib.util

    from gn_next_gen.paths import SCRIPTS

    spec = importlib.util.spec_from_file_location(
        "run_network_calibration", SCRIPTS / "run_network_calibration.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_run_is_done_only_if_it_carries_the_current_columns(tmp_path):
    """ "The file is there" is not "the file is what this code would write".

    Provenance: the first real Step 3 grid finished R1 and R2 -- the two
    *released* series -- before ``first_year`` was added to the observer table. A
    presence-only check would have skipped them on every relaunch and published
    the stale schema. Step 2 hit the same thing once and fixed it the same way.
    """
    import pandas as pd

    runner = _load_runner()
    root = tmp_path / "runs"

    assert not runner.is_done("R1", root=root)

    directory = root / "R1"
    directory.mkdir(parents=True)
    pd.DataFrame({"date_ord": [1], "gn": [1.0]}).to_parquet(
        directory / "series.parquet", index=False
    )
    (directory / "meta.json").write_text("{}")
    # Series and meta present, observers missing: not done.
    assert not runner.is_done("R1", root=root)

    stale = pd.DataFrame(
        {"station": [1], "factor": [1.0], "factor_sd": [0.1], "recovered_threshold_msh": [30.0]}
    )
    stale.to_parquet(directory / "observers.parquet", index=False)
    assert not runner.is_done("R1", root=root), "a stale schema must be redone"

    current = stale.assign(first_year=[1900])
    current.to_parquet(directory / "observers.parquet", index=False)
    assert runner.is_done("R1", root=root)


def test_the_runner_pins_the_cpu_before_importing_jax():
    """The bug that lost 101 of 102 benchmark cells to CUDA OOM.

    The pin has to happen before *anything* imports JAX, so it lives above the
    package imports rather than inside ``main``. Pinning it afterwards is a no-op
    that looks correct.
    """
    from gn_next_gen.paths import SCRIPTS

    source = (SCRIPTS / "run_network_calibration.py").read_text()
    pin = source.index('os.environ.setdefault("JAX_PLATFORMS"')
    first_package_import = source.index("from gn_next_gen")
    assert pin < first_package_import, "the device pin must precede any JAX import"
    assert "XLA_PYTHON_CLIENT_PREALLOCATE" in source


def test_the_orchestrator_exports_the_device_and_checks_progress_files():
    """A stage that achieved nothing must not be able to print "ok"."""
    from gn_next_gen.paths import SCRIPTS

    source = (SCRIPTS / "run_step3_grid.sh").read_text()
    assert "export JAX_PLATFORMS=" in source
    assert "stage_failures" in source
    # The path the failures are actually recorded at, which the first version got
    # wrong and so reported nothing.
    assert "outputs/step2/logs/baselines_progress.json" in source
    assert "outputs/step3/runs/progress.json" in source
