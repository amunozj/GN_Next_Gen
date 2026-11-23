"""Run every baseline over every generated world.

This is the long one -- roughly 3-6 hours for the full grid -- so **the PI
launches it, not the session**::

    nohup python scripts/run_baselines.py \
          --config configs/step2_baselines.yaml --jobs 8 \
          > outputs/step2/logs/baselines.log 2>&1 &

Resumable per ``(scenario, realization, method)``; a cell already on disk is
skipped unless ``--force``.  It reads only the **open** half of a realization:
``tests/test_benchmark_blindness.py`` fails if any baseline module can reach the
other one.

``--structural`` runs the S6 ensemble instead: the same S2 worlds with each
method's free graph choices perturbed one knob at a time, so the spread those
choices produce can be reported beside the parametric uncertainty.

Usage
-----
    python scripts/run_baselines.py --config configs/step2_baselines.yaml
        [--scenarios S1 S2] [--methods CEA17 UEA21] [--jobs N] [--force]
        [--structural]
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gn_next_gen.baselines import ALL_METHODS  # noqa: E402
from gn_next_gen.benchmark.progress import Progress  # noqa: E402
from gn_next_gen.benchmark.scenarios import REFERENCE_STATION  # noqa: E402
from gn_next_gen.benchmark.store import (  # noqa: E402
    BENCHMARK_ROOT,
    is_complete,
    read_observations,
    read_reference_areas,
)
from gn_next_gen.config import load_run_config, write_lineage  # noqa: E402
from gn_next_gen.graph import build_overlap_graph  # noqa: E402
from gn_next_gen.paths import OUTPUTS  # noqa: E402

BASELINE_ROOT = OUTPUTS / "step2" / "baselines"


def cell_dir(scenario: str, seed: int, *, root: Path = BASELINE_ROOT, tag: str = "") -> Path:
    name = f"seed-{seed:04d}" + (f"__{tag}" if tag else "")
    return Path(root) / scenario / name


def cell_done(scenario: str, seed: int, method: str, *, root=BASELINE_ROOT, tag="") -> bool:
    base = cell_dir(scenario, seed, root=root, tag=tag)
    return (base / f"{method}_series.parquet").is_file()


def run_cell(
    scenario: str,
    seed: int,
    method: str,
    data_root: str,
    out_root: str,
    graph_kwargs: dict,
    tag: str = "",
) -> str:
    """One (scenario, realization, method) cell, written to its own files."""
    observations = read_observations(scenario, seed, root=data_root)
    reference_areas = read_reference_areas(scenario, seed, root=data_root)
    meta = json.loads((Path(data_root) / scenario / f"seed-{seed:04d}" / "config.json").read_text())
    threshold = _anchor_threshold(meta["anchoring"])

    graph = build_overlap_graph(observations, **graph_kwargs)
    if REFERENCE_STATION not in graph.index and method not in ("UEA21", "GRP", "null-prior"):
        # Without the reference in the graph a chained method has no anchor; say
        # so rather than writing a silently unanchored series.
        raise RuntimeError("the reference observer is absent from the overlap graph")

    result = ALL_METHODS[method](
        observations,
        graph=graph,
        reference=REFERENCE_STATION,
        reference_areas=reference_areas,
        reference_threshold_msh=threshold,
        seed=seed,
    )
    base = cell_dir(scenario, seed, root=Path(out_root), tag=tag)
    base.mkdir(parents=True, exist_ok=True)
    result.series.to_parquet(base / f"{method}_series.parquet", index=False)
    result.observers.to_parquet(base / f"{method}_observers.parquet", index=False)
    (base / f"{method}_meta.json").write_text(
        json.dumps(
            {k: v for k, v in result.meta.items() if not hasattr(v, "to_parquet")},
            indent=2,
            default=str,
        )
        + "\n"
    )
    return "done"


def _anchor_threshold(anchoring: str) -> float:
    from gn_next_gen.benchmark.truth import anchor_threshold

    return anchor_threshold(anchoring)


def _discover(data_root: Path, scenarios: list[str] | None) -> list[tuple[str, int]]:
    """Every realization on disk, so the runner follows the generator."""
    found: list[tuple[str, int]] = []
    for scenario_dir in sorted(Path(data_root).glob("*")):
        if not scenario_dir.is_dir():
            continue
        if scenarios and scenario_dir.name not in scenarios:
            continue
        for seed_dir in sorted(scenario_dir.glob("seed-*")):
            seed = int(seed_dir.name.split("-")[1])
            if is_complete(scenario_dir.name, seed, root=data_root):
                found.append((scenario_dir.name, seed))
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/step2_baselines.yaml"))
    parser.add_argument("--scenarios", nargs="*", default=None)
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=None,
        help="restrict to these realizations; the scorer reads whatever is on disk, "
        "so this is how a deliberate slice is kept from mixing with a wider run",
    )
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--data-root", type=Path, default=BENCHMARK_ROOT)
    parser.add_argument("--out-root", type=Path, default=BASELINE_ROOT)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--structural", action="store_true", help="run the S6 ensemble")
    args = parser.parse_args()

    config = load_run_config(args.config)
    methods = args.methods or [*config.params["methods"], *config.params.get("nulls", [])]
    unknown = set(methods) - set(ALL_METHODS)
    if unknown:
        raise SystemExit(f"unknown method(s) {sorted(unknown)}; expected {sorted(ALL_METHODS)}")

    graph_defaults = dict(config.params.get("graph", {}))
    realizations = _discover(args.data_root, args.scenarios)
    if args.seeds:
        realizations = [cell for cell in realizations if cell[1] in set(args.seeds)]
    if not realizations:
        raise SystemExit(f"no complete realization under {args.data_root}; generate first")

    cells: list[tuple[str, int, str, dict, str]] = []
    if args.structural:
        spec = config.params["structural_ensemble"]
        source = spec["source_scenario"]
        wanted = [c for c in realizations if c[0] == source][: int(spec["n_realizations"])]
        if not wanted:
            # A long overnight run that silently does nothing is worse than one
            # that fails: the structural ensemble reuses S2's worlds, so if S2
            # has not been generated there is no work and the operator should
            # hear about it now.
            raise SystemExit(
                f"the structural ensemble needs {source} worlds and none are complete "
                f"under {args.data_root}; generate {source} first"
            )
        for knob, values in spec["perturbations"].items():
            for value in values:
                kwargs = {**graph_defaults, knob: value}
                tag = f"{knob}={value}"
                cells += [
                    (scenario, seed, method, kwargs, tag)
                    for scenario, seed in wanted
                    for method in methods
                ]
    else:
        cells = [
            (scenario, seed, method, graph_defaults, "")
            for scenario, seed in realizations
            for method in methods
        ]

    log_dir = OUTPUTS / "step2" / "logs"
    label = "structural" if args.structural else "baselines"
    progress = Progress(log_dir / f"{label}_progress.json", total=len(cells), label=label)
    write_lineage(
        config,
        log_dir,
        extra={"script": "run_baselines.py", "n_cells": len(cells), "mode": label},
        filename=f"lineage_{label}.json",
    )

    pending = [
        cell
        for cell in cells
        if args.force or not cell_done(cell[0], cell[1], cell[2], root=args.out_root, tag=cell[4])
    ]
    progress.record(skipped=len(cells) - len(pending))

    data_root, out_root = str(args.data_root), str(args.out_root)

    if args.jobs > 1:
        # run_cell must be submitted directly, not through a closure: a nested
        # function cannot be pickled, so a ProcessPoolExecutor rejects every
        # task with "Can't get local object". That failed all 189 cells of a
        # trial grid while reporting each one as an ordinary cell failure.
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = {
                pool.submit(run_cell, scenario, seed, method, data_root, out_root, kwargs, tag): (
                    scenario,
                    seed,
                    method,
                    kwargs,
                    tag,
                )
                for scenario, seed, method, kwargs, tag in pending
            }
            for future in as_completed(futures):
                _report(future, futures[future], progress)
    else:
        for cell in pending:
            scenario, seed, method, kwargs, tag = cell
            try:
                run_cell(scenario, seed, method, data_root, out_root, kwargs, tag)
            except Exception as error:  # noqa: BLE001 - one cell must not kill the grid
                progress.record(failure=f"{cell[:3]}: {error!r}")
                print(f"FAILED {cell[:3]}: {error!r}", flush=True)
            else:
                progress.record(done=1)
                print(f"   done {cell[0]} seed {cell[1]} {cell[2]} {cell[4]}", flush=True)

    print(progress.summary())


def _report(future, cell, progress: Progress) -> None:
    try:
        future.result()
    except Exception as error:  # noqa: BLE001
        progress.record(failure=f"{cell[:3]}: {error!r}")
        print(f"FAILED {cell[:3]}: {error!r}", flush=True)
    else:
        progress.record(done=1)
        print(f"   done {cell[0]} seed {cell[1]} {cell[2]} {cell[4]}", flush=True)


if __name__ == "__main__":
    main()
