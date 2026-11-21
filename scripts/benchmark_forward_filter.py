"""Measure the forward filter's cost -- Step 4's early task 1, done early.

The research report's figures for one likelihood evaluation (about 5e8 flops, ~25 MB,
0.2-1 s on CPU and 1-3 s single-chain on GPU at T = 1.52e5, K = 41) are explicitly
the researchers' estimates and were never benchmarked.  Step 4's plan says to measure
them before building on them; Check A builds the filter anyway, so it is measured here.

Usage
-----
    python scripts/benchmark_forward_filter.py [--out-dir outputs/step1/benchmark]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from gn_next_gen.inference.forward_filter import (  # noqa: E402
    count_emissions,
    forward_loglik,
    forward_loglik_two_class,
)
from gn_next_gen.paths import OUTPUTS  # noqa: E402
from gn_next_gen.viz.checks import filter_benchmark_figure  # noqa: E402
from gn_next_gen.viz.style import save_both_modes  # noqa: E402

#: 1610-2026 on a daily grid -- the full length the Step 4 series has to cover.
FULL_LENGTH = 152_000
LENGTHS = (10_000, 40_000, FULL_LENGTH)
K = 41


def time_call(fn, *args, repeats: int = 3) -> float:
    """Median wall time of a jitted call, after one compile-and-warm run."""
    value = fn(*args)
    jax.block_until_ready(value)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append(time.perf_counter() - start)
    return float(np.median(times))


def benchmark(device: str) -> list[dict]:
    rows = []
    with jax.default_device(jax.devices(device)[0]):
        rng = np.random.default_rng(0)
        for n_days in LENGTHS:
            counts = jnp.asarray(rng.integers(0, 12, n_days))
            observed = jnp.asarray(rng.random(n_days) > 0.4)
            lam = jnp.asarray(np.exp(np.sin(np.arange(n_days) / 2000.0)))
            emissions = count_emissions(counts, observed, K)

            one = jax.jit(lambda e, lam_t: forward_loglik(e, 0.88, lam_t, K))
            seconds = time_call(one, emissions, lam)
            rows.append(
                {
                    "device": device,
                    "states": K,
                    "n_days": n_days,
                    "seconds": seconds,
                    "variant": "one class",
                }
            )
            print(f"  {device} {K:>5} states T={n_days:>7,}: {seconds:7.3f} s", flush=True)

            if n_days <= 40_000:
                k2 = 41
                lam2 = jnp.stack([lam * 0.6, lam * 0.4], axis=1)
                phi2 = jnp.asarray([0.85, 0.92])
                two = jax.jit(
                    lambda c, o, lam_t, phi=phi2, k=k2: forward_loglik_two_class(
                        c, o, phi, lam_t, k
                    )
                )
                seconds = time_call(two, counts, observed, lam2, repeats=2)
                rows.append(
                    {
                        "device": device,
                        "states": k2 * k2,
                        "n_days": n_days,
                        "seconds": seconds,
                        "variant": "two classes",
                    }
                )
                print(
                    f"  {device} {k2 * k2:>5} states T={n_days:>7,}: {seconds:7.3f} s", flush=True
                )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=OUTPUTS / "step1" / "benchmark")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    devices = ["cpu"] + (["gpu"] if any(d.platform == "gpu" for d in jax.devices()) else [])
    rows = []
    for device in devices:
        print(f"{device}:")
        rows.extend(benchmark(device))

    table = pd.DataFrame(rows)
    table.to_csv(args.out_dir / "filter_benchmark.csv", index=False, lineterminator="\n")
    save_both_modes(
        lambda mode: filter_benchmark_figure(table, mode),
        args.out_dir / "figures",
        "f12_filter_benchmark",
    )

    full = table[table["n_days"] == FULL_LENGTH]
    print(f"\nAt full length (T = {FULL_LENGTH:,}, K = {K}):")
    for _, row in full.iterrows():
        print(f"  {row['device']:>3}: {row['seconds']:.3f} s per likelihood evaluation")
    print(f"\nwrote {args.out_dir}/filter_benchmark.csv")


if __name__ == "__main__":
    main()
