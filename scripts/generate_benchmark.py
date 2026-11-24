"""Generate the benchmark's synthetic worlds.

Resumable per ``(scenario, realization)``: a realization already complete on disk
is skipped unless ``--force``.  Writes ``progress.json`` beside the output so a
detached run can be watched without tailing its log.

Expect roughly 30-60 minutes for the full grid, which is past the 15-minute line
the plan draws, so **the PI launches this, not the session**::

    nohup python scripts/generate_benchmark.py \
          --config configs/step2_benchmark.yaml --jobs 8 \
          > outputs/step2/logs/generate.log 2>&1 &

Usage
-----
    python scripts/generate_benchmark.py --config configs/step2_benchmark.yaml
        [--scenarios S1 S2] [--realizations N] [--jobs N] [--force] [--check]
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gn_next_gen.benchmark.progress import Progress  # noqa: E402
from gn_next_gen.benchmark.scenarios import SCENARIOS, build_realization  # noqa: E402
from gn_next_gen.benchmark.store import (  # noqa: E402
    BENCHMARK_ROOT,
    dataset_digest,
    is_complete,
    write_dataset,
)
from gn_next_gen.config import load_run_config, write_lineage  # noqa: E402
from gn_next_gen.paths import OUTPUTS  # noqa: E402


def seeds_for(scenario: str, count: int, base: int, offsets: dict) -> list[int]:
    """Seeds are base + scenario offset + index, so counts move independently."""
    return [base + int(offsets.get(scenario, 0)) + index for index in range(count)]


def build_one(scenario_name: str, seed: int, root: str, force: bool) -> tuple[str, int, str]:
    scenario = SCENARIOS[scenario_name]
    if not force and is_complete(scenario_name, seed, root=root):
        return scenario_name, seed, "skipped"
    truth, network, config = build_realization(scenario, seed)
    write_dataset(scenario_name, seed, truth, network, config, root=root)
    return scenario_name, seed, "done"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/step2_benchmark.yaml"))
    parser.add_argument("--scenarios", nargs="*", default=None)
    parser.add_argument("--realizations", type=int, default=None, help="override the count")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--root", type=Path, default=BENCHMARK_ROOT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--check",
        action="store_true",
        help="regenerate into a scratch root and compare digests, without touching the originals",
    )
    args = parser.parse_args()

    config = load_run_config(args.config)
    planned = config.params["scenarios"]
    offsets = config.params.get("scenario_seed_offsets", {})
    names = args.scenarios or list(planned)

    jobs: list[tuple[str, int]] = []
    for name in names:
        if name not in SCENARIOS:
            raise SystemExit(f"unknown scenario {name!r}; expected {sorted(SCENARIOS)}")
        count = args.realizations or int(planned[name])
        jobs += [(name, seed) for seed in seeds_for(name, count, config.seed, offsets)]

    log_dir = OUTPUTS / "step2" / "logs"
    progress = Progress(log_dir / "generate_progress.json", total=len(jobs), label="generate")
    write_lineage(
        config,
        log_dir,
        extra={"script": "generate_benchmark.py", "n_cells": len(jobs)},
        filename="lineage_generate.json",
    )

    if args.check:
        _check(jobs, args.root, progress)
        return

    root = str(args.root)
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = {
                pool.submit(build_one, name, seed, root, args.force): (name, seed)
                for name, seed in jobs
            }
            for future in as_completed(futures):
                name, seed = futures[future]
                try:
                    _n, _s, status = future.result()
                except Exception as error:  # noqa: BLE001 - one cell must not kill the grid
                    progress.record(failure=f"{name}/{seed}: {error!r}")
                    print(f"FAILED {name} seed {seed}: {error!r}", flush=True)
                else:
                    progress.record(**{status: 1})
                    print(f"{status:>7} {name} seed {seed}", flush=True)
    else:
        for name, seed in jobs:
            try:
                _n, _s, status = build_one(name, seed, root, args.force)
            except Exception as error:  # noqa: BLE001
                progress.record(failure=f"{name}/{seed}: {error!r}")
                print(f"FAILED {name} seed {seed}: {error!r}", flush=True)
            else:
                progress.record(**{status: 1})
                print(f"{status:>7} {name} seed {seed}", flush=True)

    print(progress.summary())


def _check(jobs, root: Path, progress: Progress) -> None:
    """Determinism: rebuild into a scratch root and compare every file's digest."""
    import shutil
    import tempfile

    scratch = Path(tempfile.mkdtemp(prefix="gn-benchmark-check-"))
    try:
        mismatched = []
        for name, seed in jobs:
            if not is_complete(name, seed, root=root):
                progress.record(skipped=1)
                continue
            build_one(name, seed, str(scratch), force=True)
            original = dataset_digest(name, seed, root=root)
            rebuilt = dataset_digest(name, seed, root=scratch)
            differing = [k for k in original if original[k] != rebuilt.get(k)]
            if differing:
                mismatched.append((name, seed, differing))
                progress.record(failure=f"{name}/{seed}: {differing}")
                print(f"DIFFERS {name} seed {seed}: {differing}", flush=True)
            else:
                progress.record(done=1)
                print(f"  MATCH {name} seed {seed}", flush=True)
        print(progress.summary())
        if mismatched:
            raise SystemExit(1)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
