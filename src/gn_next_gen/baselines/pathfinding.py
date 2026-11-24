"""Pathfinding: shortest-path, multipath chaining, ported from the 2017 MATLAB code.

Called ``Backbone`` until 2026-11-24 and renamed to end a collision: **SvSc16**
is Svalgaard & Schatten's backbone, a different baseline in the same registry, so
"the backbone method" named two of the seven.  This one is the graph method -- edge
costs from error propagation, then a path -- which is what the brief
(``docs/00_brief/modernization.md``) already called a pathfinding problem.

``legacy/GN_Series_matlab/`` cannot be run: it needs MATLAB with the Curve Fitting
and Statistics toolboxes, three File Exchange functions, and the **V1-12** database,
which is not in this repository and whose station numbers do not transfer (its
``stnd = 445`` and ``endnd = 590`` are Koyama and Janssens under v1.22, where the
intended nodes are 466 and 614).  So the logic is ported and endpoints resolve by
observer name.

What is kept, because it is what the method *is*:

* robust weighted proportional fits between overlapping observers
  (``A1_Matrix_Calculation_weighted_fit.m`` l. 129-168);
* edge cost ``|sigma_a / a| / overlap_days`` (``A3_*.m`` l. 59-99);
* multiplicative chaining with relative uncertainties added in quadrature, and a
  multipath composite that removes used nodes and repeats, combining the passes
  with ``1/sigma^2`` weights (``GN_Reconstruction_MultiPath.m`` l. 507-691).

What is dropped, per ``docs/03_legacy/reuse_map.md``: the forward-in-time-only
edge direction, which stops information flowing back from later observers.

One legacy defect is **not** reproduced.  ``GN_Reconstruction_MultiPath.m``
l. 668-669 multiplies by ``LnkVlP`` a second time after ``CpmObs`` has already
been scaled, inflating the propagated uncertainty by one factor of ``a``.  With
the ``runcsvs`` reproduction dropped there is no number to match, and keeping a
bug that makes this baseline look *less* certain would flatter every method it is
compared against.  Recorded here and in the results doc.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from gn_next_gen.baselines.base import (
    CalibrationResult,
    PairEstimate,
    PairwiseEstimator,
    Series,
    chain_from_reference,
    combine_observers,
    estimate_edges,
    matched_counts,
    station_series,
)

#: Legacy gates: minimum unique (count, count) pairs and minimum Spearman rho.
MIN_UNIQUE_PAIRS = 3
MIN_RHO = 0.6

#: Passes of the multipath composite (``mp = 1:10`` in the MATLAB).
N_PASSES = 10

#: Bisquare tuning constant, the MATLAB default.
BISQUARE_C = 4.685


def robust_proportional_fit(
    target: np.ndarray, source: np.ndarray, weights: np.ndarray | None = None
) -> tuple[float, float]:
    """Weighted bisquare fit of ``target = a * source``; returns ``(a, sd(a))``.

    Iteratively reweighted least squares, which is what MATLAB's
    ``fit(..., 'Robust', 'Bisquare')`` does, implemented here because the Curve
    Fitting Toolbox is not available.
    """
    weights = np.ones_like(source, dtype=float) if weights is None else weights.astype(float)
    source = source.astype(float)
    target = target.astype(float)
    if (source**2 * weights).sum() <= 0:
        return np.nan, np.inf

    slope = float((weights * source * target).sum() / (weights * source**2).sum())
    for _ in range(10):
        residual = target - slope * source
        scale = 1.4826 * np.median(np.abs(residual - np.median(residual)))
        if scale <= 0:
            break
        normalized = residual / (BISQUARE_C * scale)
        bisquare = np.where(np.abs(normalized) < 1, (1 - normalized**2) ** 2, 0.0)
        combined = weights * bisquare
        denominator = (combined * source**2).sum()
        if denominator <= 0:
            break
        updated = float((combined * source * target).sum() / denominator)
        if abs(updated - slope) < 1e-10:
            slope = updated
            break
        slope = updated

    residual = target - slope * source
    dof = max(len(source) - 1, 1)
    variance = float((weights * residual**2).sum() / dof / max((weights * source**2).sum(), 1e-12))
    return slope, float(np.sqrt(max(variance, 0.0)))


class PathfindingPair(PairwiseEstimator):
    """One edge of the pathfinding graph: a robust proportional link."""

    name = "Pathfinding"

    def estimate(self, left, right, prepared_left, prepared_right):
        del prepared_left, prepared_right
        return _pathfinding_pair(left, right)


def _pathfinding_pair(left: Series, right: Series) -> PairEstimate:
    target, source = matched_counts(left, right)
    if len(target) < MIN_UNIQUE_PAIRS:
        return PairEstimate(np.nan, np.inf, len(target), usable=False)

    # Collapse to unique (count, count) pairs with frequency weights, as the
    # MATLAB does with sub2ind/unique/histc.
    pairs, frequency = np.unique(np.stack([source, target], axis=1), axis=0, return_counts=True)
    if len(pairs) < MIN_UNIQUE_PAIRS:
        return PairEstimate(np.nan, np.inf, len(target), usable=False)

    rho = spearmanr(source, target).statistic
    if not np.isfinite(rho) or rho < MIN_RHO:
        return PairEstimate(np.nan, np.inf, len(target), usable=False)

    slope, sd = robust_proportional_fit(pairs[:, 1], pairs[:, 0], frequency.astype(float))
    if not np.isfinite(slope) or slope <= 0:
        return PairEstimate(np.nan, np.inf, len(target), usable=False)
    return PairEstimate(
        factor=slope,
        relative_sd=float(sd / slope) if slope > 0 else np.inf,
        n_days=int(len(target)),
        usable=True,
    )


def calibrate_pathfinding(
    observations: pd.DataFrame,
    graph,
    reference: int,
    *,
    n_passes: int = N_PASSES,
    **_kwargs,
) -> CalibrationResult:
    """Multipath composite: chain, record, drop the used nodes, repeat.

    Each pass is a single shortest-path chain from the reference.  Removing the
    observers a pass used forces the next pass onto different links, and the
    passes are combined with ``1/sigma^2`` weights -- the legacy composite's
    structure, including its assumption that the passes are independent when they
    in fact share edges.  That assumption is one of the things the coverage figure
    is here to catch.
    """
    series = station_series(observations)
    estimates = estimate_edges(graph, series, PathfindingPair())

    available = dict(estimates)
    composites, all_scales = [], []
    used: set[int] = set()
    for _pass in range(n_passes):
        scales = chain_from_reference(graph, available, reference)
        reached = scales[scales["reached"] & (scales["station"] != reference)]
        if reached.empty:
            break
        daily = combine_observers(observations, scales)
        if daily.empty:
            break
        composites.append(daily)
        all_scales.append(scales)

        used |= set(reached["station"].astype(int))
        available = {
            key: value
            for key, value in available.items()
            if key[0] not in used and key[1] not in used
        }
        if not available:
            break

    if not composites:
        return CalibrationResult(
            pd.DataFrame(columns=["date_ord", "gn", "gn_sd", "n_observers"]),
            pd.DataFrame(),
            "Pathfinding",
            {"reason": "no usable path from the reference"},
        )

    stacked = pd.concat(composites, ignore_index=True)
    stacked["weight"] = 1.0 / np.maximum(stacked["gn_sd"].to_numpy(), 1e-3) ** 2
    grouped = stacked.groupby("date_ord")
    series_out = pd.DataFrame(
        {
            "date_ord": grouped["gn"].mean().index.to_numpy(),
            "gn": grouped.apply(
                lambda b: float(np.average(b["gn"], weights=b["weight"])),
                include_groups=False,
            ).to_numpy(),
            "gn_sd": np.sqrt(1.0 / grouped["weight"].sum().to_numpy()),
            "n_observers": grouped["n_observers"].max().to_numpy(),
        }
    )

    observers = (
        pd.concat(all_scales, ignore_index=True)
        .sort_values("n_hops")
        .drop_duplicates("station", keep="first")
        .assign(recovered_threshold_msh=np.nan)
    )
    return CalibrationResult(
        series=series_out,
        observers=observers,
        name="Pathfinding",
        meta={"n_passes": len(composites), "n_edges_estimated": len(estimates) // 2},
    )
