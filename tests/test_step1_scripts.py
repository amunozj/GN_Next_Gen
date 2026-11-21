"""The Step 1 scripts must run end to end and produce what the results doc cites.

Marked slow: each one fits or bootstraps over real data. The point is not the
numbers -- those are checked where they are computed -- but that the scripts stay
runnable, keep writing the CSVs the figures and the results doc read, and keep
honouring the firewall by writing a lineage record.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from gn_next_gen.paths import CONFIGS, REPO_ROOT, SCRIPTS


def _run(script: str, *args: str, cwd=REPO_ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script), *args],
        capture_output=True,
        text=True,
        cwd=cwd,
        env={"JAX_PLATFORMS": "cpu", **_env()},
    )


def _env() -> dict:
    import os

    return {k: v for k, v in os.environ.items() if k != "JAX_PLATFORMS"}


@pytest.mark.slow
def test_check_a_runs_and_writes_its_outputs(tmp_path):
    proc = _run(
        "check_a_dynamics.py",
        "--quick",
        "--config",
        str(CONFIGS / "step1_check_a.yaml"),
        "--out-dir",
        str(tmp_path),
    )
    assert proc.returncode == 0, proc.stderr[-3000:]

    for name in (
        "scores.csv",
        "acf.csv",
        "zero_by_bin.csv",
        "count_pmf.csv",
        "run_lengths.csv",
        "lineage.json",
    ):
        assert (tmp_path / name).is_file(), name
    figures = {p.name for p in (tmp_path / "figures").glob("*.png")}
    assert any(n.startswith("f07") for n in figures)
    assert any(n.startswith("f09") for n in figures)
    assert all(
        f"{stem}_{mode}.png" in figures
        for stem in {n.rsplit("_", 1)[0] for n in figures}
        for mode in ("light", "dark")
    )

    import pandas as pd

    scores = pd.read_csv(tmp_path / "scores.csv")
    assert {"survey", "threshold_msh", "n_classes", "phi", "aggregate"} <= set(scores.columns)
    assert (scores["phi"] > 0).all() and (scores["phi"] < 1).all()


@pytest.mark.slow
def test_check_b_runs_and_writes_its_outputs(tmp_path):
    proc = _run(
        "check_b_stationarity.py",
        "--quick",
        "--config",
        str(CONFIGS / "step1_check_b.yaml"),
        "--out-dir",
        str(tmp_path),
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    for name in ("percentiles.csv", "epoch_pairs.csv", "mixtures.csv", "lineage.json"):
        assert (tmp_path / name).is_file(), name


@pytest.mark.slow
def test_every_run_records_its_lineage_and_the_firewall(tmp_path):
    """ADR-002 wants lineage per run; a run that does not write it is not compliant."""
    proc = _run(
        "check_b_stationarity.py",
        "--quick",
        "--config",
        str(CONFIGS / "step1_check_b.yaml"),
        "--out-dir",
        str(tmp_path),
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    record = json.loads((tmp_path / "lineage.json").read_text())
    assert record["variant"] == "sunspot_only"
    assert record["firewall"]["proxy_derived_calibration_inputs"] == []
    assert record["seed"]
    assert {entry["path"] for entry in record["inputs"]}


def test_scripts_are_importable_without_side_effects():
    """Importing a script must not run an analysis or touch outputs/."""
    import importlib.util

    for name in (
        "check_a_dynamics",
        "check_b_stationarity",
        "make_step1_figures",
        "diff_gn_releases",
        "fetch_silso",
        "benchmark_forward_filter",
    ):
        spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        assert hasattr(module, "main"), name
