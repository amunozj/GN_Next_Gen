"""Step 3 on the real record: both released anchorings, and the sensitivities.

**The PI launches this, not a Claude Code session** (Step 2 decision 10, carried
forward): it is hours, and a session dies on a disconnect.  Resumable per run, so
an interrupted grid continues rather than restarting::

    nohup python scripts/run_network_calibration.py \\
          --config configs/step3_network.yaml --jobs 8 \\
          > outputs/step3/logs/network.log 2>&1 &

Usage
-----
    python scripts/run_network_calibration.py --config configs/step3_network.yaml
        [--runs R1 R2] [--jobs N] [--force] [--quick] [--list]

``--quick`` fits one anchoring over a short modern window with a small bootstrap.
It is the smoke test a session may run; it is not a result.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Pin the device **before** anything imports JAX.  JAX takes the GPU by default
# and preallocates most of its memory, so N parallel workers cannot share four
# A100s: every one dies with CUDA_ERROR_OUT_OF_MEMORY, which is how the first real
# grid lost 101 of 102 benchmark cells while the stage still reported "ok".  CPU
# is also the right default on merit -- Step 1 measured the GPU 4.8x slower at
# this shape and `check_a_dynamics.py` pins itself to the CPU for the same reason.
# `--device gpu` exists to be measured, not assumed.
if "--device" in sys.argv:
    _device = sys.argv[sys.argv.index("--device") + 1]
else:
    _device = os.environ.get("GN_DEVICE", "cpu")
if _device == "cpu":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gn_next_gen.benchmark.progress import Progress  # noqa: E402
from gn_next_gen.config import load_run_config, write_lineage  # noqa: E402
from gn_next_gen.graph import (  # noqa: E402
    build_overlap_graph,
    observer_diagnostics,
    resistance_by_year,
)
from gn_next_gen.io import load_observations  # noqa: E402
from gn_next_gen.models import emissions as em  # noqa: E402
from gn_next_gen.models import network as nw  # noqa: E402
from gn_next_gen.models.activity_map import fit_activity_map  # noqa: E402
from gn_next_gen.paths import OUTPUTS  # noqa: E402

STEP3_ROOT = OUTPUTS / "step3"
RUN_ROOT = STEP3_ROOT / "runs"


def run_plan(config) -> list[dict]:
    """Every run this config asks for: the released series, then the sensitivities."""
    params = config.params
    model = dict(params["model"])
    runs: list[dict] = []
    for index, entry in enumerate(params["released"], start=1):
        runs.append(
            {
                "name": f"R{index}",
                "anchoring": entry["anchoring"],
                "label": entry["label"],
                "released": True,
                **model,
            }
        )
    base = runs[0]["anchoring"]
    sensitivities = params.get("sensitivities", {})

    if "anchor_extension" in sensitivities:
        entry = sensitivities["anchor_extension"]
        runs.append(
            {
                "name": "R3",
                "anchoring": entry["anchoring"],
                "label": f"sensitivity vs {entry['against']}",
                "released": False,
                **model,
            }
        )
    for key, field in (
        ("drift_prior", "drift_log_sigma_mean"),
        ("fingerprint_ablation", "fingerprint_weight"),
        ("smoothness", "lambda_penalty"),
        ("day_tolerance", "day_tolerance"),
    ):
        for value in sensitivities.get(key, {}).get(field, []):
            runs.append(
                {
                    "name": f"S-{key}-{field}={value}",
                    "anchoring": base,
                    "label": f"{field} = {value}",
                    "released": False,
                    **model,
                    field: value,
                }
            )
    for value in sensitivities.get("activity_tilt", {}).get("exponent_scale", []):
        runs.append(
            {
                "name": f"S-tilt-{value}",
                "anchoring": base,
                "label": f"activity exponents x {value}",
                "released": False,
                "exponent_scale": value,
                **model,
            }
        )
    return runs


def run_directory(name: str, *, root: Path = RUN_ROOT) -> Path:
    return Path(root) / name.replace("/", "_")


#: Columns a *current* run's observer table carries.  "The file is there" is not
#: the same as "the file is what this code would write": a run finished before a
#: column existed has the files but not the information, and a resumable grid
#: would skip it forever.  Step 2 hit this exactly once and fixed it the same way
#: -- ``run_baselines.cell_done`` refuses a fit whose parquet lacks the
#: uncertainty-decomposition columns -- and checking the schema costs one footer
#: read with no version bookkeeping.
#:
#: Provenance, because it is the reason this exists: the first real grid completed
#: R1 and R2 before ``first_year`` was added to the table, so the two *released*
#: series would have been published from a stale schema.
REQUIRED_OBSERVER_COLUMNS = frozenset(
    {"station", "factor", "factor_sd", "recovered_threshold_msh", "first_year"}
)


def is_done(name: str, *, root: Path = RUN_ROOT) -> bool:
    """Whether this run is on disk *and* carries what the figures and scorer read."""
    directory = run_directory(name, root=root)
    if not (directory / "series.parquet").is_file():
        return False
    if not (directory / "meta.json").is_file():
        return False
    observers = directory / "observers.parquet"
    if not observers.is_file():
        return False
    try:
        import pyarrow.parquet as pq

        columns = set(pq.read_schema(observers).names)
    except Exception:  # noqa: BLE001 -- an unreadable run is a run to redo
        return False
    return REQUIRED_OBSERVER_COLUMNS <= columns


def execute(spec: dict, config, *, root: Path, quick: bool = False) -> str:
    """One calibration run, written to its own directory."""
    started = time.time()
    anchoring = spec["anchoring"]
    params = config.params
    observation_options = dict(params["observations"])
    graph_options = dict(params["graph"])
    if "day_tolerance" in spec:
        graph_options["day_tolerance"] = spec["day_tolerance"]
    uncertainty = dict(params["uncertainty"])

    anchor_rows, threshold = nw.anchor_observations(anchoring)
    observations = load_observations(
        release=config.release,
        duplicate_rule=config.duplicate_rule,
        kmas_role=config.kmas_role,
        include_extension=observation_options.get("include_extension", True),
        extension_threshold=observation_options.get("extension_threshold", "anchor"),
        extension_post_v13_only=observation_options.get("extension_post_v13_only", False),
    )
    if quick:
        cutoff = anchor_rows["date_ord"].min() - 365 * 20
        observations = observations[observations["date_ord"] >= cutoff]
    network = pd.concat(
        [
            observations[["date_ord", "station", "groups"]],
            anchor_rows[["date_ord", "station", "groups"]],
        ],
        ignore_index=True,
    )

    survey = nw.ANCHORINGS[anchoring]["survey"]
    exponents = None
    if "exponent_scale" in spec:
        exponents = {
            key: value * float(spec["exponent_scale"])
            for key, value in em.ACTIVITY_EXPONENTS.items()
        }
    detection = em.build_detection_grid(
        survey,
        reference_threshold_msh=threshold,
        exponents=exponents,
        source="empirical",
    )
    activity_map = fit_activity_map(survey, threshold)

    graph = build_overlap_graph(network, **graph_options)
    fit = nw.calibrate_network(
        network,
        detection=detection,
        activity_map=activity_map,
        anchor_station=nw.ANCHOR_STATION,
        name=spec["name"],
        kernel=spec.get("kernel", "hybrid"),
        graph=graph,
        fingerprint_weight=float(spec.get("fingerprint_weight", 1.0)),
        lambda_penalty=float(spec.get("lambda_penalty", nw.LAMBDA_PENALTY)),
        drift_log_sigma_mean=float(spec.get("drift_log_sigma_mean", nw.DRIFT_LOG_SIGMA_MEAN)),
        n_outer=int(spec.get("n_outer", 4)),
        n_bootstrap=20 if quick else int(uncertainty["n_bootstrap"]),
        n_draws=40 if quick else int(uncertainty["n_draws"]),
        maxiter=200 if quick else int(spec.get("maxiter", 600)),
        seed=config.seed,
    )

    directory = run_directory(spec["name"], root=root)
    directory.mkdir(parents=True, exist_ok=True)
    fit.series.to_parquet(directory / "series.parquet", index=False)
    fit.observers.to_parquet(directory / "observers.parquet", index=False)

    slots = pd.DataFrame(
        {
            "station": fit.inputs.slot_station,
            "year": fit.inputs.slot_year,
            "n_days": np.asarray(fit.static["slot_n_days"]),
        }
    )
    import jax.numpy as jnp

    log_r, log_threshold = nw.log_rate_ratio(
        jnp.asarray(fit.theta), fit.inputs, fit.layout, fit.static
    )
    slots["log_r"] = np.asarray(log_r)
    slots["threshold_msh"] = np.exp(np.asarray(log_threshold))
    slots.to_parquet(directory / "slots.parquet", index=False)

    # Graph diagnostics, per anchoring and between them.
    anchors = {anchoring: [nw.ANCHOR_STATION]}
    diagnostics = observer_diagnostics(graph, anchors)
    diagnostics.to_parquet(directory / "diagnostics.parquet", index=False)
    resistance_by_year(graph, [nw.ANCHOR_STATION]).to_parquet(
        directory / "resistance_by_year.parquet", index=False
    )

    meta = {
        **fit.meta,
        "run": spec["name"],
        "anchoring": anchoring,
        "label": spec["label"],
        "released": bool(spec["released"]),
        "threshold_msh": threshold,
        "graph": graph_options,
        "quick": bool(quick),
        "seconds": time.time() - started,
        "n_observations": int(len(network)),
        "exponent_scale": spec.get("exponent_scale"),
    }
    (directory / "meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    write_lineage(config, directory, extra={"run": spec["name"], "anchoring": anchoring})
    return "done"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/step3_network.yaml")
    parser.add_argument("--runs", nargs="*", default=None)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--out", default=str(RUN_ROOT))
    parser.add_argument(
        "--device",
        default=os.environ.get("GN_DEVICE", "cpu"),
        choices=("cpu", "gpu"),
        help="cpu by default: the GPU is 4.8x slower at this shape (Step 1) and "
        "parallel workers cannot share it",
    )
    args = parser.parse_args(argv)

    config = load_run_config(args.config)
    plan = run_plan(config)
    if args.runs:
        plan = [spec for spec in plan if spec["name"] in set(args.runs)]
    if args.list:
        for spec in plan:
            print(f"{spec['name']:34s} {spec['anchoring']:16s} {spec['label']}")
        return 0

    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    progress = Progress(path=root / "progress.json", total=len(plan), label="step3 runs")
    progress.write()

    pending = [spec for spec in plan if args.force or not is_done(spec["name"], root=root)]
    for spec in plan:
        if spec not in pending:
            progress.record(skipped=1)
    print(f"{len(pending)} of {len(plan)} runs to do", flush=True)

    failures: list[str] = []
    if args.jobs <= 1:
        for spec in pending:
            try:
                execute(spec, config, root=root, quick=args.quick)
                progress.record(done=1)
            except Exception as error:  # noqa: BLE001 -- report and carry on
                failures.append(f"{spec['name']}: {error}")
                progress.record(failure=f"{spec['name']}: {error}")
            print(progress.summary(), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = {
                pool.submit(execute, spec, config, root=root, quick=args.quick): spec
                for spec in pending
            }
            for future in as_completed(futures):
                spec = futures[future]
                try:
                    future.result()
                    progress.record(done=1)
                except Exception as error:  # noqa: BLE001
                    failures.append(f"{spec['name']}: {error}")
                    progress.record(failure=f"{spec['name']}: {error}")
                print(progress.summary(), flush=True)

    if failures:
        print("\nFAILURES:", flush=True)
        for line in failures:
            print(f"  {line}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
