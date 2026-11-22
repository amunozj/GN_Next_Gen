"""Score every baseline against the sealed truth, and write the tables.

Short (~10 minutes over the full grid) because the work was done upstream; it
reads the sealed half, which nothing under ``baselines`` may do.

Outputs, all under ``outputs/step2/scores/``:

``scores.csv``
    one row per (scenario, realization, method) -- the 42-cell table.
``coverage.csv``
    empirical against nominal coverage at 50/68/90/95%, pooled per method.
``case_study_observers.csv``
    the ten named observers, true against recovered acuity, per method.
``structural.csv``
    the S6 ensemble, when ``--structural`` outputs exist.

Usage
-----
    python scripts/score_benchmark.py --config configs/step2_benchmark.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gn_next_gen.baselines.base import CalibrationResult  # noqa: E402
from gn_next_gen.benchmark.observers import CASE_STUDY_KINDS  # noqa: E402
from gn_next_gen.benchmark.progress import Progress  # noqa: E402
from gn_next_gen.benchmark.scoring import COVERAGE_LEVELS, score  # noqa: E402
from gn_next_gen.benchmark.store import (  # noqa: E402
    BENCHMARK_ROOT,
    read_sealed_parameters,
    read_sealed_truth,
)
from gn_next_gen.config import load_run_config, write_lineage  # noqa: E402
from gn_next_gen.paths import OUTPUTS  # noqa: E402

BASELINE_ROOT = OUTPUTS / "step2" / "baselines"
SCORE_ROOT = OUTPUTS / "step2" / "scores"


def _cells(baseline_root: Path, tagged: bool) -> list[tuple[str, int, str, str, Path]]:
    found = []
    for scenario_dir in sorted(Path(baseline_root).glob("*")):
        if not scenario_dir.is_dir():
            continue
        for seed_dir in sorted(scenario_dir.glob("seed-*")):
            name = seed_dir.name
            tag = name.split("__", 1)[1] if "__" in name else ""
            if bool(tag) != tagged:
                continue
            seed = int(name.split("-")[1].split("__")[0])
            for series in sorted(seed_dir.glob("*_series.parquet")):
                method = series.name.removesuffix("_series.parquet")
                found.append((scenario_dir.name, seed, method, tag, seed_dir))
    return found


def _load(seed_dir: Path, method: str) -> CalibrationResult:
    observers_path = seed_dir / f"{method}_observers.parquet"
    return CalibrationResult(
        series=pd.read_parquet(seed_dir / f"{method}_series.parquet"),
        observers=pd.read_parquet(observers_path) if observers_path.is_file() else pd.DataFrame(),
        name=method,
    )


def _case_study_rows(result: CalibrationResult, parameters: pd.DataFrame, scenario, seed):
    """True against recovered acuity for the ten named observers (f25/f26)."""
    named = parameters[parameters["kind"].isin(CASE_STUDY_KINDS)]
    if named.empty or result.observers.empty:
        return []
    merged = named.merge(result.observers, on="station", how="left")
    rows = []
    for row in merged.itertuples():
        rows.append(
            {
                "scenario": scenario,
                "seed": seed,
                "method": result.name,
                "station": int(row.station),
                "kind": row.kind,
                "true_threshold_msh": float(row.base_threshold_msh),
                "recovered_threshold_msh": float(getattr(row, "recovered_threshold_msh", np.nan)),
                "factor": float(getattr(row, "factor", np.nan)),
                "reached": bool(getattr(row, "reached", False)),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/step2_benchmark.yaml"))
    parser.add_argument("--data-root", type=Path, default=BENCHMARK_ROOT)
    parser.add_argument("--baseline-root", type=Path, default=BASELINE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=SCORE_ROOT)
    parser.add_argument("--structural", action="store_true")
    args = parser.parse_args()

    config = load_run_config(args.config)
    cells = _cells(args.baseline_root, tagged=args.structural)
    if not cells:
        raise SystemExit(f"no baseline output under {args.baseline_root}; run the baselines first")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    label = "score-structural" if args.structural else "score"
    progress = Progress(
        OUTPUTS / "step2" / "logs" / f"{label}_progress.json", total=len(cells), label=label
    )
    write_lineage(
        config, args.out_dir, extra={"script": "score_benchmark.py", "n_cells": len(cells)}
    )

    truths: dict[tuple[str, int], tuple] = {}
    rows, case_rows = [], []
    for scenario, seed, method, tag, seed_dir in cells:
        key = (scenario, seed)
        if key not in truths:
            truths[key] = (
                read_sealed_truth(scenario, seed, root=args.data_root),
                read_sealed_parameters(scenario, seed, root=args.data_root),
            )
        truth, parameters = truths[key]
        try:
            result = _load(seed_dir, method)
            row = score(truth, result, parameters, scenario=scenario, seed=seed)
            row["perturbation"] = tag
            rows.append(row)
            case_rows += _case_study_rows(result, parameters, scenario, seed)
        except Exception as error:  # noqa: BLE001
            progress.record(failure=f"{scenario}/{seed}/{method}: {error!r}")
            print(f"FAILED {scenario} {seed} {method}: {error!r}", flush=True)
        else:
            progress.record(done=1)

    scores = pd.DataFrame(rows)
    name = "structural.csv" if args.structural else "scores.csv"
    scores.to_csv(args.out_dir / name, index=False, lineterminator="\n")
    print(f"wrote {args.out_dir / name} ({len(scores)} cells)")

    if not args.structural:
        coverage = _coverage_table(scores)
        coverage.to_csv(args.out_dir / "coverage.csv", index=False, lineterminator="\n")
        print(f"wrote {args.out_dir / 'coverage.csv'}")

        if case_rows:
            cases = pd.DataFrame(case_rows)
            cases.to_csv(
                args.out_dir / "case_study_observers.csv", index=False, lineterminator="\n"
            )
            print(f"wrote {args.out_dir / 'case_study_observers.csv'} ({len(cases)} rows)")

        print()
        print(_headline(scores).to_string(index=False))

    print(progress.summary())


def _coverage_table(scores: pd.DataFrame) -> pd.DataFrame:
    columns = [f"coverage_{int(level * 100)}" for level in COVERAGE_LEVELS]
    present = [c for c in columns if c in scores]
    grouped = scores.groupby(["scenario", "method"])[present].mean().reset_index()
    melted = grouped.melt(id_vars=["scenario", "method"], var_name="level", value_name="empirical")
    melted["nominal"] = melted["level"].str.removeprefix("coverage_").astype(float) / 100
    return melted[["scenario", "method", "nominal", "empirical"]].sort_values(
        ["scenario", "method", "nominal"], ignore_index=True
    )


def _headline(scores: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "bias_yearly",
        "crmse_yearly",
        "amplitude_slope",
        "coverage_50",
        "coverage_90",
        "crps_yearly",
    ]
    present = [c for c in keep if c in scores]
    return scores.groupby("method")[present].mean().round(3).reset_index()


if __name__ == "__main__":
    main()
