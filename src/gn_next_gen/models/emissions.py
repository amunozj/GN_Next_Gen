"""What an observer reports, given what the anchor would have counted.

The estimand is the anchor survey's own reference-equivalent count: the number of
groups the anchor would have recorded above its own completeness threshold
(ADR-003, ADR-009).  Every other observer is a *relative* detector of that
quantity, and this module is the map between the two.

The detection ratio
-------------------
::

    k_o(t) = fraction_above(T_o(t); law(AL_t)) / fraction_above(T_ref; law(AL_t))

``k_o < 1`` for an observer coarser than the anchor and ``k_o > 1`` for a finer
one, which really happens -- Debrecen's threshold is 20.4 MSH and RGO's 28.1, so
each is the finer instrument under the other's anchoring.

**Why the activity level is in there at all.**  Step 2 measured the published
acuity methods' error and found it was not a per-observer mistake: recovered
acuity tracks how active the Sun truly was over that observer's own career, at
Spearman -0.62 (UEA21) and -0.73 (GRP), because both match an observer's
statistic against a library built **once** over the anchor epoch.  "An observer
whose era is quieter than cycles 15-19 therefore has a statistic below *every*
library entry, the fit runs monotonically to the top of the grid, and it pins at
the 300 MSH cap" (Step 2 results, f23).  The ubercal derivation names the fix in
survey-calibration terms: "A scalar k-factor is FGCM's gray approximation; **a
detection function applied to an area distribution that changes with activity is
the chromatic correction**."

Three things are taken from frozen upstream or measured Step 1 results and are
**not refitted here**, which is what ADR-011 requires:

1. the per-survey Weibull + log-normal parameters (``area_law_2026-11``);
2. the *exponents* of the activity dependence, from the 2015 published power laws
   -- Step 1 reproduced KMAS's ``mu`` exponent to three decimals (0.021) and its
   ``sigma`` exponent at 0.109 against 0.117 (results §2.2);
3. the per-group to per-group-day shift in ``mu``, measured in Step 1 §2.4.
   This one is easy to miss and it points the "wrong" way: the per-day law is
   **smaller**, by about 0.2 in log area, because the published law uses each
   group's *peak* area while a detection integral sees instantaneous areas,
   including every small day at the start and end of a group's life.  Step 1 says
   plainly that "the published per-group law may not be used for ``q_o``
   unmodified".

The emission table
------------------
``P(y | N, k, s)`` in two stages, so that thinning, augmentation and splitting are
separate mechanisms rather than one fudged mean::

    V | N ~ Binomial(N, k)                 if k <= 1   (the observer misses groups)
    V | N = N + Poisson(N (k - 1))         if k > 1    (the observer splits what the
                                                        anchor merged, or sees deeper)
    y | V = V + Binomial(V, s)                         (splitting convention)

``E[y | N] = k N (1 + s)``, but ``k`` and ``s`` are not interchangeable: the split
rate changes the *shape* at fixed mean, which is what identifies it separately.
With ``k = 1`` and ``s = 0`` the table is the identity, so an observer declared
identical to the anchor reports the estimand exactly -- the property the benchmark's
reference observer needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import binom, poisson

from gn_next_gen.io.cache import cached_arrays, cached_frame
from gn_next_gen.io.reference import load_day_status, load_group_single_entry
from gn_next_gen.paths import (
    AREA_LAW_POSTERIOR,
    REFERENCE_DAY_STATUS,
    REFERENCE_GROUP_DAY,
)
from gn_next_gen.reference.area_law import (
    ACTIVITY_SIGMA_DAYS,
    MUHEM_PER_MHEM,
    PUBLISHED_2015,
    AreaLawParams,
    activity_level,
    fraction_above,
    gaussian_smooth,
    load_upstream_parameters,
)

#: Truncation of the reported-count axis, matching Step 1's state space.
#: **D17**: never size this from ``max(GROUPS)`` -- v1.3's two largest values are
#: implausible outliers above a tail that stops at 53.
MAX_COUNT = 41

#: Activity exponents of the log-normal component, from the 2015 published power
#: laws (``PUBLISHED_2015``).  Only the exponents are used: the amplitudes come
#: from each survey's own upstream calibration, so the law passes through
#: upstream's values at that survey's reference activity.  Step 1 measured the
#: amplitudes as "systematically below the published ones" for two identified
#: reasons (a different weight convention and a higher truncation), neither of
#: them a discrepancy to tune away -- which is exactly why the amplitudes are not
#: taken from the paper.
ACTIVITY_EXPONENTS = {
    "mu_LN": PUBLISHED_2015["mu_LN"][1],
    "sigma_LN": PUBLISHED_2015["sigma_LN"][1],
    "c_LN": PUBLISHED_2015["c_LN"][1],
}

#: Per-group to per-group-day shift in ``mu``, measured in Step 1 §2.4 (results
#: table: RGO -0.194, Debrecen -0.197, KMAS -0.262).  Applied because ``k_o``
#: integrates over the areas **on the disk on a day**, while the frozen law
#: describes one peak area per group.
GRAIN_SHIFT_MU = {"rgo": -0.194, "debrecen": -0.197, "kmas": -0.262}

#: Fallback for a survey Step 1 did not measure a shift for (Schwabe, Spörer,
#: which D23 kept out of Step 1).  The mean of the three measured values, carried
#: explicitly so a run can see it was a fallback rather than a measurement.
GRAIN_SHIFT_DEFAULT = float(np.mean(list(GRAIN_SHIFT_MU.values())))

#: Detection-ratio grid.  Log-spaced because ``k`` is a ratio and the likelihood
#: is smooth in ``log k``; the span covers an observer 50x coarser than the anchor
#: up to one 5x finer, which brackets every real observer by a wide margin.
N_K_GRID = 96
K_GRID_LO, K_GRID_HI = 0.02, 5.0

#: Split-rate grid for the emission library.
SPLIT_GRID = np.array([0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50])


def k_grid(n: int = N_K_GRID) -> np.ndarray:
    """The detection-ratio grid the emission library is built on."""
    return np.exp(np.linspace(np.log(K_GRID_LO), np.log(K_GRID_HI), n))


# ---------------------------------------------------------------------------
# The area law at an activity level
# ---------------------------------------------------------------------------


def params_at_activity(
    base: AreaLawParams,
    activity: float | np.ndarray,
    activity_ref: float,
    *,
    exponents: dict[str, float] | None = None,
    grain_shift: float | None = None,
) -> AreaLawParams | list[AreaLawParams]:
    """The frozen law rescaled to an activity level.

    ``mu``, ``sigma`` and the log-normal weight ``c`` scale as powers of
    ``activity / activity_ref``; the Weibull parameters do **not** move, which is
    Step 1's measured result (§2.1: no dependence detectable in any survey, and
    "the signs disagree across surveys, which is what noise looks like rather than
    a trend"), and it is what licenses holding ``(k, lambda)`` fixed.

    ``grain_shift`` is added to ``mu`` to convert the per-group law to the
    per-group-day law a detection integral needs (§2.4).
    """
    exponents = ACTIVITY_EXPONENTS if exponents is None else exponents
    if grain_shift is None:
        grain_shift = GRAIN_SHIFT_MU.get(base.survey, GRAIN_SHIFT_DEFAULT)

    scalar = np.isscalar(activity)
    ratio = np.atleast_1d(np.asarray(activity, dtype=float)) / float(activity_ref)
    ratio = np.where(np.isfinite(ratio) & (ratio > 0), ratio, 1.0)

    out = []
    for value in ratio:
        mu = base.mu_LN * value ** exponents["mu_LN"] + grain_shift
        sigma = base.sigma_LN * value ** exponents["sigma_LN"]
        c = base.c_LN * value ** exponents["c_LN"]
        out.append(
            base.replace(
                mu_LN=float(mu),
                sigma_LN=float(max(sigma, 1e-3)),
                pi_W=float(np.clip(1.0 - c, 1e-6, 1.0 - 1e-6)),
            )
        )
    return out[0] if scalar else out


def reference_activity(survey: str, *, use_cache: bool = True) -> float:
    """The activity level at which a survey's frozen law is taken to hold.

    The median over the survey's own active days, with the **instantaneous**
    weighting -- one scale shared by :class:`Catalogue`, the activity map and this,
    because the two weightings differ by about six and mixing them would misplace
    every observer on the grid's activity axis.  Scaling from here means the law
    reproduces upstream's calibrated parameters at the survey's typical activity
    and only moves away from them as the cycle does.
    """
    frame = cached_frame(
        "reference_activity",
        [REFERENCE_DAY_STATUS, AREA_LAW_POSTERIOR],
        {"survey": survey, "weighting": "instantaneous"},
        lambda: _reference_activity_frame(survey),
        use_cache=use_cache,
    )
    return float(frame["activity_ref"].iloc[0])


def _reference_activity_frame(survey: str) -> pd.DataFrame:
    day_status = load_day_status([survey])
    series = activity_level(day_status, weighting="instantaneous")
    observed = day_status.loc[day_status["day_status"] > 0, "date_ord"]
    values = series.reindex(observed).dropna()
    return pd.DataFrame(
        {
            "survey": [survey],
            "activity_ref": [float(np.median(values))],
            "n_days": [int(len(values))],
        }
    )


# ---------------------------------------------------------------------------
# The detection ratio, as a grid a JAX likelihood can read
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DetectionGrid:
    """``log k_o`` on a grid of ``(log threshold, activity)``.

    ``fraction_above`` is a quadrature over the latent area law, so it is not
    something to call inside an optimizer loop a few thousand times.  It is
    tabulated once here and read by bilinear interpolation, which keeps the single
    source of truth for the area law in ``reference.area_law`` and keeps the
    likelihood smooth and differentiable in ``log T_o``.

    Attributes
    ----------
    log_threshold
        Grid in ``log(T_o / MSH)``.
    activity
        Grid in activity level (mHem).
    log_k
        ``(n_threshold, n_activity)`` table of ``log k_o``.
    """

    log_threshold: np.ndarray
    activity: np.ndarray
    log_k: np.ndarray
    survey: str
    reference_threshold_msh: float
    activity_ref: float
    meta: dict = field(default_factory=dict)

    def __call__(self, threshold_msh, activity) -> np.ndarray:
        """Bilinear ``log k`` at arbitrary thresholds and activity levels (NumPy)."""
        return np.asarray(
            bilinear(
                self.log_threshold,
                self.activity,
                self.log_k,
                np.log(np.asarray(threshold_msh, dtype=float)),
                np.asarray(activity, dtype=float),
            )
        )


def bilinear(x_grid, y_grid, table, x, y):
    """Bilinear interpolation on a regular-in-index grid, clamped at the edges.

    Written with plain arithmetic so the same function serves NumPy and JAX
    arrays; the likelihood differentiates through it with respect to ``x``.
    """
    xp = _as_module(x)
    ix = xp.clip(xp.searchsorted(x_grid, x) - 1, 0, len(x_grid) - 2)
    iy = xp.clip(xp.searchsorted(y_grid, y) - 1, 0, len(y_grid) - 2)
    x0, x1 = x_grid[ix], x_grid[ix + 1]
    y0, y1 = y_grid[iy], y_grid[iy + 1]
    tx = xp.clip((x - x0) / (x1 - x0), 0.0, 1.0)
    ty = xp.clip((y - y0) / (y1 - y0), 0.0, 1.0)
    c00 = table[ix, iy]
    c10 = table[ix + 1, iy]
    c01 = table[ix, iy + 1]
    c11 = table[ix + 1, iy + 1]
    return c00 * (1 - tx) * (1 - ty) + c10 * tx * (1 - ty) + c01 * (1 - tx) * ty + c11 * tx * ty


def _as_module(value):
    """NumPy or jax.numpy, whichever ``value`` belongs to."""
    module = type(value).__module__
    if module.startswith("jaxlib") or module.startswith("jax"):
        import jax.numpy as jnp

        return jnp
    return np


#: Fewest group-days above a threshold before an empirical ratio is trusted in an
#: activity bin.  Below it the cell falls back to the rescaled parametric law.
MIN_GROUPS_PER_CELL = 30

#: Activity bins the empirical table is built on, as quantiles of the survey's own
#: active days so every bin holds data.
N_ACTIVITY_BINS = 12


@dataclass(frozen=True)
class Catalogue:
    """A per-group-day area catalogue and the activity level it implies.

    Two paths produce one of these and the rest of Step 3 cannot tell them apart,
    which is the point: the real record's frozen reference survey, and a benchmark
    world's own open ``reference_areas``.  A synthetic world has its own areas, so
    a method calibrated on it must measure detection from *those* -- using the
    frozen catalogue there would be reading the answer off the wrong Sun.

    ``activity`` is the **instantaneous** weighting -- every group present on the
    day, smoothed -- because that is the one both paths can supply, and because it
    is "the natural covariate for the per-group-*day* area law" that a detection
    integral works on (``activity_level``'s docstring).  Note that the published
    power-law *exponents* were measured on the emergence weighting; only the ratio
    ``AL / AL_ref`` enters :func:`params_at_activity`, and the parametric branch is
    a fallback outside the catalogue's coverage, so the mismatch is second order.
    It is recorded here rather than left implicit.
    """

    date_ord: np.ndarray
    area_msh: np.ndarray
    activity: pd.Series
    name: str = "catalogue"

    @property
    def n_group_days(self) -> int:
        return len(self.area_msh)


def catalogue_from_survey(survey: str, *, sigma_days: float = ACTIVITY_SIGMA_DAYS) -> Catalogue:
    """The frozen reference survey as a :class:`Catalogue`."""
    from gn_next_gen.io.reference import load_group_days

    day_status = load_day_status([survey])
    groups = load_group_days([survey])
    activity = activity_level(day_status, weighting="instantaneous", sigma_days=sigma_days)
    return Catalogue(
        date_ord=groups["date_ord"].to_numpy(dtype=np.int64),
        area_msh=groups["area_corr"].to_numpy(dtype=float),
        activity=activity,
        name=survey,
    )


def catalogue_from_areas(
    reference_areas: pd.DataFrame,
    *,
    area_column: str = "area_msh",
    sigma_days: float = ACTIVITY_SIGMA_DAYS,
    name: str = "reference_areas",
) -> Catalogue:
    """A benchmark world's open ``reference_areas`` as a :class:`Catalogue`.

    The activity level is computed from the same table, by summing every group
    present on a day and smoothing -- the instantaneous weighting, which is all a
    group-day table can supply and what the detection integral wants anyway.
    """
    frame = reference_areas[["date_ord", area_column]].dropna()
    daily = frame.groupby("date_ord")[area_column].sum()
    days = daily.index.to_numpy(dtype=np.int64)
    lo, hi = int(days.min()), int(days.max())
    grid = np.arange(lo, hi + 1)
    dense = np.full(grid.shape, np.nan)
    dense[days - lo] = daily.to_numpy()
    smoothed = gaussian_smooth(dense, sigma_days) / MUHEM_PER_MHEM
    return Catalogue(
        date_ord=frame["date_ord"].to_numpy(dtype=np.int64),
        area_msh=frame[area_column].to_numpy(dtype=float),
        activity=pd.Series(smoothed, index=grid, name="activity_level"),
        name=name,
    )


def empirical_detection_table(
    catalogue: Catalogue,
    thresholds: np.ndarray,
    *,
    reference_threshold_msh: float,
    n_bins: int = N_ACTIVITY_BINS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``log k`` measured directly in an area catalogue, binned by activity.

    For a set of group-days, ``k(T)`` is the number with ``area > T`` divided by the
    number with ``area > T_ref``.  That is ``fraction_above`` evaluated
    non-parametrically on the catalogue itself, and it is **already at
    per-group-day grain**, so the per-group shift of Step 1 §2.4 does not apply --
    that correction exists to convert the published per-group law, and here there
    is no conversion to make.

    Measured against the parametric law on RGO, the law under-predicts ``k`` for
    observers coarser than the anchor by 4% at 40 MSH rising to 13% at 200 MSH, so
    this table removes a real bias in ``k_o``, which multiplies every one of that
    observer's counts (PI, 2026-11-24).

    Returns ``(log_k, activity_centres, n_above)``; cells backed by fewer than
    :data:`MIN_GROUPS_PER_CELL` group-days are ``nan``.
    """
    areas = np.asarray(catalogue.area_msh, dtype=float)
    group_activity = catalogue.activity.reindex(catalogue.date_ord).to_numpy()
    usable = np.isfinite(group_activity) & np.isfinite(areas)
    areas = areas[usable]
    group_activity = group_activity[usable]
    if len(areas) < MIN_GROUPS_PER_CELL:
        raise ValueError(f"{catalogue.name}: too few group-days for an empirical table")

    edges = np.unique(np.quantile(group_activity, np.linspace(0.0, 1.0, n_bins + 1)))
    bin_index = np.clip(np.searchsorted(edges, group_activity, side="right") - 1, 0, len(edges) - 2)
    n_bins_actual = len(edges) - 1
    centres = np.array(
        [
            float(np.median(group_activity[bin_index == b])) if np.any(bin_index == b) else np.nan
            for b in range(n_bins_actual)
        ]
    )

    log_k = np.full((len(thresholds), n_bins_actual), np.nan)
    n_above = np.zeros((len(thresholds), n_bins_actual), dtype=np.int64)
    for b in range(n_bins_actual):
        in_bin = areas[bin_index == b]
        if len(in_bin) == 0:
            continue
        denominator = int(np.count_nonzero(in_bin > reference_threshold_msh))
        if denominator < MIN_GROUPS_PER_CELL:
            continue
        for i, threshold in enumerate(thresholds):
            numerator = int(np.count_nonzero(in_bin > threshold))
            n_above[i, b] = numerator
            if numerator >= MIN_GROUPS_PER_CELL:
                log_k[i, b] = np.log(numerator) - np.log(denominator)

    exact = np.isclose(thresholds, reference_threshold_msh)
    log_k[exact, :] = np.where(np.isfinite(centres), 0.0, np.nan)
    return log_k, centres, n_above


def detection_grid(
    survey: str,
    *,
    source: str = "empirical",
    use_cache: bool = True,
    force: bool = False,
    **kwargs,
) -> DetectionGrid:
    """:func:`build_detection_grid` for the frozen reference, cached in ``data/interim/``.

    Building the grid costs about 4 s -- a quadrature over the area law at every
    (threshold, activity) node, plus a pass over the catalogue -- which is nothing
    once and a real cost spread over a few hundred resumable workers.  Keyed on the
    raw inputs' manifest checksums, so a grid built from another release cannot be
    read by mistake.

    Only for the real record: a benchmark world passes its own ``catalogue``, which
    has no manifest row and so cannot be keyed this way.
    """
    if (
        kwargs.get("thresholds") is not None
        or kwargs.get("activities") is not None
        or kwargs.get("catalogue") is not None
    ):
        return build_detection_grid(survey, source=source, **kwargs)

    options = {
        "survey": survey,
        "source": source,
        **{k: v for k, v in sorted(kwargs.items()) if v is not None},
    }

    def build() -> dict:
        grid = build_detection_grid(survey, source=source, **kwargs)
        return {
            "log_threshold": grid.log_threshold,
            "activity": grid.activity,
            "log_k": grid.log_k,
            "reference_threshold_msh": np.asarray(grid.reference_threshold_msh),
            "activity_ref": np.asarray(grid.activity_ref),
            "meta": grid.meta,
        }

    payload = cached_arrays(
        "detection_grid",
        [REFERENCE_DAY_STATUS, REFERENCE_GROUP_DAY, AREA_LAW_POSTERIOR],
        options,
        build,
        use_cache=use_cache,
        force=force,
    )
    return DetectionGrid(
        log_threshold=np.asarray(payload["log_threshold"]),
        activity=np.asarray(payload["activity"]),
        log_k=np.asarray(payload["log_k"]),
        survey=survey,
        reference_threshold_msh=float(payload["reference_threshold_msh"]),
        activity_ref=float(payload["activity_ref"]),
        meta=dict(payload.get("meta", {})),
    )


def build_detection_grid(
    survey: str,
    *,
    catalogue: Catalogue | None = None,
    reference_threshold_msh: float | None = None,
    thresholds: np.ndarray | None = None,
    activities: np.ndarray | None = None,
    activity_ref: float | None = None,
    exponents: dict[str, float] | None = None,
    grain_shift: float | None = None,
    n_grid: int = 2048,
    source: str = "empirical",
    n_activity_bins: int = N_ACTIVITY_BINS,
) -> DetectionGrid:
    """Tabulate ``log k_o`` for one anchor survey.

    Parameters
    ----------
    survey
        The anchor survey, whose frozen law and completeness threshold define the
        estimand (ADR-009).
    reference_threshold_msh
        ``T_ref``.  Defaults to the survey's own ``threshold_MSH`` from the frozen
        calibration table, which is what makes ``k_ref = 1`` exactly.
    source
        ``"empirical"`` (default, PI 2026-11-24) measures ``k`` in the reference
        catalogue wherever it has coverage and uses the parametric law **only to
        extrapolate** below and above that coverage, rescaled per threshold so the
        two branches agree at the boundary.  ``"parametric"`` uses the law
        everywhere, which is what the approved plan specified and what the
        ``S-tilt`` sensitivity perturbs.
    exponents, grain_shift
        Overridable so the ``S-tilt`` sensitivity can perturb them.  SDSS's worst
        systematic came from a term "held fixed rather than fitted", so the
        consequence of holding these is measured rather than assumed.  Note that
        under ``source="empirical"`` they steer only the extrapolation branch --
        the threshold dependence comes from the data and the activity dependence
        beyond coverage from the law.

    Notes
    -----
    Why the split is between *threshold* and *activity* dependence.  The catalogue
    measures the threshold dependence exactly, on the grain the detection integral
    needs.  What it cannot do is reach activity levels it never saw, and the
    Maunder Minimum is exactly that -- so the law supplies the activity dependence
    out there, anchored to the data at the edge of coverage.  Where the catalogue
    has nothing to say, something parametric has to speak, and this records which
    of the two is speaking in ``meta``.
    """
    base = load_upstream_parameters(survey)
    if reference_threshold_msh is None:
        reference_threshold_msh = float(base.threshold_msh)
    if source == "empirical" and catalogue is None:
        catalogue = catalogue_from_survey(survey)
    if activity_ref is None:
        activity_ref = (
            float(np.nanmedian(catalogue.activity.to_numpy()))
            if catalogue is not None
            else reference_activity(survey)
        )
    if thresholds is None:
        thresholds = np.exp(np.linspace(np.log(0.5), np.log(600.0), 160))
    # T_ref must be a grid node, not something interpolated to.  log k(T_ref) is
    # exactly 0 at every activity level by construction, and that zero is what
    # pins the absolute scale (ADR-003); interpolating near it returned 0.9988
    # instead of 1, which would be a 0.1% scale error injected at the anchor.
    thresholds = np.unique(
        np.concatenate([np.asarray(thresholds, dtype=float), [float(reference_threshold_msh)]])
    )
    if activities is None:
        top = 4.0 * float(activity_ref)
        if catalogue is not None:
            observed = catalogue.activity.to_numpy()
            if np.any(np.isfinite(observed)):
                top = max(top, 1.2 * float(np.nanmax(observed)))
        activities = np.linspace(0.002, top, 64)
    activities = np.asarray(activities, dtype=float)

    parametric = np.empty((len(thresholds), len(activities)))
    for j, activity in enumerate(activities):
        params = params_at_activity(
            base,
            float(activity),
            activity_ref,
            exponents=exponents,
            grain_shift=grain_shift,
        )
        denominator = fraction_above(reference_threshold_msh, params, n_grid=n_grid)
        for i, threshold in enumerate(thresholds):
            numerator = fraction_above(float(threshold), params, n_grid=n_grid)
            parametric[i, j] = np.log(max(numerator, 1e-300)) - np.log(max(denominator, 1e-300))

    meta = {
        "source": source,
        "exponents": dict(exponents or ACTIVITY_EXPONENTS),
        "grain_shift": float(
            grain_shift
            if grain_shift is not None
            else GRAIN_SHIFT_MU.get(survey, GRAIN_SHIFT_DEFAULT)
        ),
        "n_threshold": len(thresholds),
        "n_activity": len(activities),
    }

    if source == "parametric":
        log_k = parametric
        meta["coverage_lo"] = float("nan")
        meta["coverage_hi"] = float("nan")
        meta["empirical_fraction"] = 0.0
    elif source == "empirical":
        log_k, meta_empirical = _blend_empirical(
            catalogue,
            thresholds,
            activities,
            parametric,
            reference_threshold_msh=float(reference_threshold_msh),
            n_activity_bins=n_activity_bins,
        )
        meta.update(meta_empirical)
    else:
        raise ValueError(f"unknown source {source!r}; expected 'empirical' or 'parametric'")

    return DetectionGrid(
        log_threshold=np.log(thresholds),
        activity=activities,
        log_k=log_k,
        survey=survey,
        reference_threshold_msh=float(reference_threshold_msh),
        activity_ref=float(activity_ref),
        meta=meta,
    )


def _blend_empirical(
    catalogue: Catalogue,
    thresholds: np.ndarray,
    activities: np.ndarray,
    parametric: np.ndarray,
    *,
    reference_threshold_msh: float,
    n_activity_bins: int,
) -> tuple[np.ndarray, dict]:
    """Empirical table inside the catalogue's activity coverage, law outside it.

    The law is offset **per threshold** so that it agrees with the measurement at
    the edge of coverage.  That makes ``log k`` continuous across the boundary by
    construction rather than by luck, and it means the extrapolation inherits the
    catalogue's threshold dependence and only borrows the law's activity
    dependence -- which is the only thing the catalogue cannot supply.
    """
    empirical, centres, n_above = empirical_detection_table(
        catalogue,
        thresholds,
        reference_threshold_msh=reference_threshold_msh,
        n_bins=n_activity_bins,
    )
    good_bins = np.flatnonzero(np.isfinite(centres))
    if len(good_bins) < 2:
        raise ValueError(f"{catalogue.name}: too few usable activity bins for an empirical table")
    centres = centres[good_bins]
    empirical = empirical[:, good_bins]
    order = np.argsort(centres)
    centres = centres[order]
    empirical = empirical[:, order]
    lo, hi = float(centres[0]), float(centres[-1])

    # Fill a threshold row's missing cells from its own finite neighbours, so a
    # sparse high-threshold bin does not punch a hole in an otherwise good row.
    for i in range(empirical.shape[0]):
        row = empirical[i]
        finite = np.isfinite(row)
        if finite.sum() == 0:
            continue
        empirical[i] = np.interp(centres, centres[finite], row[finite])

    # Per-threshold offsets that pin the law to the measurement at each edge.
    parametric_at_centres = np.empty_like(empirical)
    for i in range(len(thresholds)):
        parametric_at_centres[i] = np.interp(centres, activities, parametric[i])
    offset_lo = empirical[:, 0] - parametric_at_centres[:, 0]
    offset_hi = empirical[:, -1] - parametric_at_centres[:, -1]

    out = np.empty_like(parametric)
    inside = (activities >= lo) & (activities <= hi)
    for i in range(len(thresholds)):
        row_finite = np.isfinite(empirical[i])
        if row_finite.sum() >= 2:
            out[i, inside] = np.interp(
                activities[inside], centres[row_finite], empirical[i][row_finite]
            )
        else:
            out[i, inside] = parametric[i, inside] + offset_lo[i]
        out[i, activities < lo] = parametric[i, activities < lo] + offset_lo[i]
        out[i, activities > hi] = parametric[i, activities > hi] + offset_hi[i]

    # The anchor identity is exact whatever the blending did.
    exact = np.isclose(thresholds, reference_threshold_msh)
    out[exact, :] = 0.0

    meta = {
        "coverage_lo": lo,
        "coverage_hi": hi,
        "empirical_fraction": float(inside.mean()),
        "n_activity_bins": int(len(centres)),
        "min_groups_per_cell": MIN_GROUPS_PER_CELL,
        "median_abs_law_offset": float(np.median(np.abs(offset_lo))),
    }
    return out, meta


# ---------------------------------------------------------------------------
# Emission tables
# ---------------------------------------------------------------------------


def emission_matrix(k: float, split_rate: float = 0.0, *, max_count: int = MAX_COUNT) -> np.ndarray:
    """``P(y | N)`` as a row-stochastic ``(max_count + 1, max_count + 1)`` matrix.

    Rows are the latent reference-equivalent count ``N``, columns the reported
    count ``y``.  Mass that would land above ``max_count`` is accumulated in the
    last column rather than dropped, so every row sums to one and the table never
    silently leaks probability -- the trap Step 2 recorded for composed PMF
    matrices ("composing PMF matrices leaks mass and biases the chained mean
    **low**").
    """
    size = int(max_count) + 1
    states = np.arange(size)

    # Stage 1: how many groups the observer sees at all.
    if k <= 1.0:
        visible = binom.pmf(states[None, :], states[:, None], min(max(k, 0.0), 1.0))
    else:
        extra = poisson.pmf(
            states[None, :] - states[:, None], np.maximum(states[:, None], 0) * (k - 1.0)
        )
        visible = np.nan_to_num(extra, nan=0.0)
        visible[0, 0] = 1.0
    visible = np.where(np.isfinite(visible), visible, 0.0)

    # Stage 2: splitting, which lets y exceed N.
    if split_rate <= 0.0:
        table = visible
    else:
        split = binom.pmf(states[None, :] - states[:, None], states[:, None], float(split_rate))
        split = np.nan_to_num(split, nan=0.0)
        split[0, 0] = 1.0
        table = visible @ split

    # Fold the lost upper tail into the last column; never renormalize silently.
    deficit = 1.0 - table.sum(axis=1)
    table[:, -1] += np.maximum(deficit, 0.0)
    row_sum = table.sum(axis=1, keepdims=True)
    return table / np.where(row_sum > 0, row_sum, 1.0)


def log_emission_library(
    *,
    k_values: np.ndarray | None = None,
    split_values: np.ndarray | None = None,
    max_count: int = MAX_COUNT,
    floor: float = 1e-300,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``log P(y | N)`` over a grid of ``(k, split rate)``.

    Returns ``(log_table, k_values, split_values)`` with ``log_table`` shaped
    ``(n_k, n_split, max_count + 1, max_count + 1)``.  The whole library is about
    8 MB, which is why the pairwise kernel can afford to look up rather than
    rebuild.
    """
    k_values = k_grid() if k_values is None else np.asarray(k_values, dtype=float)
    split_values = SPLIT_GRID if split_values is None else np.asarray(split_values, dtype=float)
    size = int(max_count) + 1
    table = np.empty((len(k_values), len(split_values), size, size))
    for i, k in enumerate(k_values):
        for j, split in enumerate(split_values):
            table[i, j] = emission_matrix(float(k), float(split), max_count=max_count)
    return np.log(np.maximum(table, floor)), k_values, split_values


def empirical_count_pmf(
    survey: str,
    *,
    reference_threshold_msh: float | None = None,
    n_bins: int = N_ACTIVITY_BINS,
    max_count: int = MAX_COUNT,
    use_cache: bool = True,
    force: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """The reference's own daily count distribution, binned by activity level.

    Returns ``(pmf, activity_centres)`` with ``pmf`` shaped
    ``(n_bins, max_count + 1)``, each row a distribution over the anchor's
    reference-equivalent daily count on days at that activity level.

    **Why this is measured rather than assumed Poisson.**  The occupancy
    ("fingerprint") term is the only thing that can calibrate an observer with no
    usable overlap, and it works by comparing the *shape* of a record's count
    distribution against what an observer of a given acuity would have produced.
    With a Poisson latent count that comparison is empty: thinning a Poisson gives
    another Poisson, so the shape carries no information about the thinning rate at
    all.  The research notes state the degeneracy exactly -- "multiplying all
    ``p_o`` by ``c`` and dividing ``lambda_t`` by ``c`` leaves the binomial-Poisson
    marginal unchanged, because Poisson thinning gives ``Poisson(p*lambda)``".

    The reference's real distribution is **not** Poisson -- Step 1 measured its
    dispersion index and count PMF against a fitted immigration-death chain -- so
    the term has content exactly to the extent of that departure.  How much
    content is a measured question, which is what the plan's ``w`` ablation
    answers rather than asserts.
    """
    if reference_threshold_msh is None:
        reference_threshold_msh = float(load_upstream_parameters(survey).threshold_msh)

    def build() -> dict:
        from gn_next_gen.io.reference import daily_counts

        day_status = load_day_status([survey])
        single = load_group_single_entry([survey])
        activity = activity_level(day_status, single, weighting="emergence")
        counts = daily_counts(float(reference_threshold_msh), [survey])
        day_activity = activity.reindex(counts["date_ord"].to_numpy()).to_numpy()
        values = np.clip(counts["n_groups"].to_numpy(), 0, max_count)
        usable = np.isfinite(day_activity)
        values = values[usable]
        day_activity = day_activity[usable]

        edges = np.unique(np.quantile(day_activity, np.linspace(0.0, 1.0, n_bins + 1)))
        index = np.clip(np.searchsorted(edges, day_activity, side="right") - 1, 0, len(edges) - 2)
        n_actual = len(edges) - 1
        pmf = np.zeros((n_actual, max_count + 1))
        centres = np.zeros(n_actual)
        for b in range(n_actual):
            selected = values[index == b]
            if len(selected) == 0:
                pmf[b] = 1.0 / (max_count + 1)
                centres[b] = np.nan
                continue
            histogram = np.bincount(selected, minlength=max_count + 1).astype(float)
            pmf[b] = histogram / histogram.sum()
            centres[b] = float(np.median(day_activity[index == b]))
        keep = np.isfinite(centres)
        return {"pmf": pmf[keep], "activity": centres[keep]}

    payload = cached_arrays(
        "reference_count_pmf",
        [REFERENCE_DAY_STATUS, REFERENCE_GROUP_DAY, AREA_LAW_POSTERIOR],
        {
            "survey": survey,
            "threshold": round(float(reference_threshold_msh), 6),
            "n_bins": n_bins,
            "max_count": max_count,
        },
        build,
        use_cache=use_cache,
        force=force,
    )
    return np.asarray(payload["pmf"]), np.asarray(payload["activity"])
