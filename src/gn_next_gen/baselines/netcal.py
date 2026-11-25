"""Step 3's methods, in the one shape the Step 2 scorer already reads.

Two entries, sharing one likelihood and differing only in which pairwise kernel
they use:

``NetCal``
    Kernel A everywhere, Kernel B below the low-count threshold.  The step's
    product.
``NetCal-M``
    Kernel B everywhere.  The cross-check arm, kept because the equating section's
    standing advice is to "keep a Haberman-style linking of separately estimated
    pairwise calibrations as a diagnostic" and to "treat disagreement between the
    two as a misspecification signal".

**Everything is measured from the world under test, nothing from the frozen
reference.**  A benchmark realization has its own Sun: its own areas, its own
activity history, its own anchor calendar.  The detection grid and the activity
map are therefore built from that world's open ``reference_areas`` and its
anchor's own counts.  Borrowing the frozen RGO catalogue here would be reading an
answer off a different record, and would also be the exact mistake Step 2 measured
in UEA21 and GRP -- matching an observer against a library built over *another*
epoch, which is what makes their recovered acuity track era activity at
rho = -0.62 and -0.73.

This module reads only the open half of a realization.
``tests/test_benchmark_blindness.py`` fails if it so much as names the sealed one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gn_next_gen.baselines.base import CalibrationResult
from gn_next_gen.models import emissions as em
from gn_next_gen.models import network as nw
from gn_next_gen.models.activity_map import fit_activity_map_from_series

#: Threshold grid the detection table is tabulated on, in MSH.  Wider than the
#: published grids (A157 caps at 300 msd = 150 MSH) so that a pinned estimate is a
#: finding rather than an artifact of the cap -- Step 2 found both acuity methods
#: pinned at their ceiling for quiet-era observers.
THRESHOLD_GRID = np.exp(np.linspace(np.log(1.0), np.log(500.0), 120))


def law_survey_for(threshold_msh: float) -> str:
    """Which frozen survey's law supplies the parametric extrapolation branch.

    Inferred from the anchor threshold rather than passed in, so the method runs
    through Step 2's existing ``run_baselines.py`` with no change to its signature:
    ``--methods NetCal NetCal-M`` is the whole integration.  The anchor threshold
    *is* a survey's own completeness floor (ADR-009), so it identifies the survey.
    """
    from gn_next_gen.io.reference import CORE_FIVE
    from gn_next_gen.reference.area_law import load_upstream_parameters

    best, distance = "rgo", np.inf
    for survey in CORE_FIVE:
        try:
            own = float(load_upstream_parameters(survey).threshold_msh)
        except KeyError:
            continue
        gap = abs(own - float(threshold_msh))
        if gap < distance:
            best, distance = survey, gap
    return best


def _prepare(
    observations: pd.DataFrame,
    reference_areas: pd.DataFrame,
    reference_threshold_msh: float,
    reference_station: int,
    *,
    area_column: str = "area_msh",
    survey: str = "benchmark",
    law_survey: str = "rgo",
):
    """Detection grid and activity map, both measured on this world's own data."""
    catalogue = em.catalogue_from_areas(reference_areas, area_column=area_column, name=survey)
    detection = em.build_detection_grid(
        survey,
        catalogue=catalogue,
        law_survey=law_survey,
        reference_threshold_msh=float(reference_threshold_msh),
        thresholds=THRESHOLD_GRID,
        source="empirical",
    )
    anchor = observations[observations["station"] == reference_station]
    counts = pd.DataFrame(
        {
            "date_ord": anchor["date_ord"].to_numpy(),
            "n_groups": anchor["groups"].to_numpy(),
        }
    )
    activity_map = fit_activity_map_from_series(
        counts,
        catalogue.activity,
        survey=survey,
        threshold_msh=float(reference_threshold_msh),
    )
    return detection, activity_map, catalogue


def calibrate_netcal(
    observations: pd.DataFrame,
    *,
    graph=None,
    reference: int = nw.ANCHOR_STATION,
    reference_areas: pd.DataFrame | None = None,
    reference_threshold_msh: float = 30.0,
    seed: int = 0,
    kernel: str = "hybrid",
    name: str = "NetCal",
    metadata: pd.DataFrame | None = None,
    fingerprint_weight: float = 1.0,
    lambda_penalty: float = nw.LAMBDA_PENALTY,
    drift_log_sigma_mean: float = nw.DRIFT_LOG_SIGMA_MEAN,
    n_outer: int = 3,
    n_bootstrap: int = 100,
    n_draws: int = 200,
    maxiter: int = 500,
    correct: bool = True,
    survey: str = "benchmark",
    law_survey: str | None = None,
    area_column: str = "area_msh",
    **_kwargs,
) -> CalibrationResult:
    """Run the concurrent network calibration and return the scorer's shape."""
    if reference_areas is None or reference_areas.empty:
        raise ValueError(
            "the network calibration needs the anchor's area catalogue to measure "
            "detection from; it cannot borrow the frozen reference"
        )
    if law_survey is None:
        law_survey = law_survey_for(reference_threshold_msh)
    detection, activity_map, catalogue = _prepare(
        observations,
        reference_areas,
        reference_threshold_msh,
        reference,
        area_column=area_column,
        survey=survey,
        law_survey=law_survey,
    )
    fit = nw.calibrate_network(
        observations,
        detection=detection,
        activity_map=activity_map,
        anchor_station=int(reference),
        name=name,
        kernel=kernel,
        metadata=metadata,
        graph=graph,
        catalogue=catalogue,
        fingerprint_weight=fingerprint_weight,
        lambda_penalty=lambda_penalty,
        drift_log_sigma_mean=drift_log_sigma_mean,
        n_outer=n_outer,
        n_bootstrap=n_bootstrap,
        n_draws=n_draws,
        maxiter=maxiter,
        correct=correct,
        seed=seed,
    )

    observers = fit.observers.copy()
    for column in ("factor", "factor_sd", "recovered_threshold_msh", "n_hops", "reached"):
        if column not in observers.columns:
            observers[column] = np.nan
    return CalibrationResult(
        series=fit.series,
        observers=observers,
        name=name,
        meta={**fit.meta, "passes": fit.passes, "seed": seed},
    )


def calibrate_netcal_marginal(observations: pd.DataFrame, **kwargs) -> CalibrationResult:
    """``NetCal-M``: Kernel B everywhere, the misspecification cross-check."""
    kwargs.setdefault("name", "NetCal-M")
    kwargs["kernel"] = "marginal"
    return calibrate_netcal(observations, **kwargs)
