"""The seven ported calibration baselines, and the registry the runner reads.

Five come from Chatzistergos et al. (2025), so this benchmark is a strict superset
of the closest published precedent: their five methods, plus the GRPSSN
count-distribution variant and the 2017 MATLAB multipath pathfinding method whose
daisy-chaining they named as unevaluated future work.

Two nulls are registered beside them.  A benchmark that only ranks real methods
cannot say whether the ranking means anything; a method that fails to beat a
random segmentation or a returned prior has not demonstrated skill.

Every baseline has the same signature::

    calibrate(observations, *, graph, reference, reference_areas,
              reference_threshold_msh, seed) -> CalibrationResult

and reads only the open half of a realization.  ``tests/test_benchmark_blindness.py``
fails if any module here so much as names the sealed one.
"""

from __future__ import annotations

from collections.abc import Callable

from gn_next_gen.baselines.acuity import calibrate_acuity
from gn_next_gen.baselines.base import CalibrationResult, PairEstimate
from gn_next_gen.baselines.linear import (
    calibrate_hosc98,
    calibrate_linear,
    calibrate_svsc16,
    calibrate_vhs24,
)
from gn_next_gen.baselines.netcal import (
    calibrate_netcal,
    calibrate_netcal_marginal,
)
from gn_next_gen.baselines.nulls import (
    calibrate_naive_mean,
    calibrate_naive_median,
    calibrate_prior_null,
    calibrate_random_segmentation,
)
from gn_next_gen.baselines.pathfinding import calibrate_pathfinding
from gn_next_gen.baselines.pmf import calibrate_cea17


def _uea21(observations, *, reference_areas, reference_threshold_msh, seed=0, **_kwargs):
    return calibrate_acuity(
        observations,
        reference_areas,
        reference_threshold_msh,
        fingerprint="adf",
        name="UEA21",
        seed=seed,
    )


def _grp(observations, *, reference_areas, reference_threshold_msh, seed=0, **_kwargs):
    return calibrate_acuity(
        observations,
        reference_areas,
        reference_threshold_msh,
        fingerprint="counts",
        name="GRP",
        seed=seed,
    )


def _hosc98(observations, *, graph, reference, **_kwargs):
    return calibrate_hosc98(observations, graph, reference)


def _svsc16(observations, *, graph, reference, **_kwargs):
    return calibrate_svsc16(observations, graph, reference)


def _vhs24(observations, *, graph, reference, **_kwargs):
    return calibrate_vhs24(observations, graph, reference)


def _cea17(observations, *, graph, reference, seed=0, **_kwargs):
    return calibrate_cea17(observations, graph, reference, seed=seed)


def _pathfinding(observations, *, graph, reference, **_kwargs):
    return calibrate_pathfinding(observations, graph, reference)


#: Name -> calibrator.  Order is the order every table and figure uses: the three
#: linear scalings, the two non-parametric methods, the count-distribution
#: variant, the pathfinding chain, then the nulls.
BASELINES: dict[str, Callable[..., CalibrationResult]] = {
    "HoSc98": _hosc98,
    "SvSc16": _svsc16,
    "VHS24": _vhs24,
    "CEA17": _cea17,
    "UEA21": _uea21,
    "GRP": _grp,
    "Pathfinding": _pathfinding,
    # Step 3: the concurrent fit, and its cross-check arm.
    "NetCal": calibrate_netcal,
    "NetCal-M": calibrate_netcal_marginal,
}

#: Floors, not methods.  Reported beside the baselines and never ranked with them.
NULLS: dict[str, Callable[..., CalibrationResult]] = {
    "null-prior": calibrate_prior_null,
    "null-segmentation": calibrate_random_segmentation,
    "naive-mean": calibrate_naive_mean,
    "naive-median": calibrate_naive_median,
}

ALL_METHODS = {**BASELINES, **NULLS}

__all__ = [
    "ALL_METHODS",
    "BASELINES",
    "NULLS",
    "CalibrationResult",
    "PairEstimate",
    "calibrate_acuity",
    "calibrate_naive_mean",
    "calibrate_naive_median",
    "calibrate_netcal",
    "calibrate_netcal_marginal",
    "calibrate_pathfinding",
    "calibrate_cea17",
    "calibrate_hosc98",
    "calibrate_linear",
    "calibrate_svsc16",
    "calibrate_vhs24",
]
