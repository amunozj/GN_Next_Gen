"""Calibrate every observer at once, over the whole overlap graph.

This is Approach 2 of the assessment: replace backbone and shortest-path chaining
with **one concurrent fit** of all observer parameters.  The thing being replaced
is explicit in the code it replaces -- ``baselines.base.chain_from_reference`` runs
Dijkstra and carries the anchor's scale along a single cheapest path, which is
"precisely the chained-calibration limiting case the joint model generalises, and
precisely why its error grows with effective resistance".

Why concurrent wins, in one line of algebra that is the project's own derivation
rather than a quoted result: in log space each overlap is ``y_ij = a_i - a_j + e``,
the weighted least-squares covariance is the pseudo-inverse of the graph Laplacian,
so ``Var(a_i - a_j)`` is the effective resistance, parallel paths combine like
parallel resistors, and a global solution is therefore never worse than the best
single chain.  SDSS measured the practical size of this: ubercalibration roughly
halved the scatter of the chain it replaced, whose "errors were dominated by the
Photometric Telescope (PT) based calibration method".

The three terms
---------------
**Pairwise, Kernel A (``pair_conditional``).**  With ``y_o,t ~ Poisson(r_o(t) L_t)``,
conditioning on a matched pair's total eliminates the latent daily level exactly::

    y_a | (y_a + y_b) = n  ~  BetaBinom(n, r_a / (r_a + r_b), kappa(dt))

"a Bradley-Terry likelihood with ``L_t`` eliminated, so overlap days are literally
match results" (*derived*).  No prior on the latent count, no nuisance to profile,
and the Fisher information of the resulting system **is** the graph Laplacian, so
the resistance diagnostics describe the fit rather than merely accompanying it.

``kappa(dt)`` is the lifetime-based decorrelation the spec asks for, in place of the
+/-2-day "simultaneous" assumption that ``reuse_map.md`` puts on the Drop list.  It
comes from the reference's **own measured** autocorrelation (Step 1 f07, with its
13-day transit trough and 27-day recurrence bump): at ``dt = 0`` the kernel is a
plain binomial, and a lag is honestly overdispersed rather than silently equated.

**What Kernel A can and cannot identify.**  It constrains the *rate ratio*
``r_o = k_o * (1 + s_o)`` and not ``k_o`` and the split rate ``s_o`` separately,
because a Bradley-Terry likelihood sees only rates.  That is not a defect to hide:
splitting and coarse acuity genuinely produce the same mean, and only the *shape*
of the count distribution tells them apart.  Kernel B and the occupancy term are
what carry that shape, and the ``w``-ablation measures how much it is worth.

**Pairwise, Kernel B (``pair_marginal``).**  The exact marginalization,

::

    P(y_b | y_a, dt) = sum_N P(y_b | N, theta_b) P(N | y_a, dt, theta_a)

over the emission tables of :mod:`gn_next_gen.models.emissions`.  It exists because
the conditional-binomial reduction is documented to break "for nonlinear threshold
effects ... and **at low activity, where zeros dominate**" -- which is the Maunder
Minimum, the era this step has to reach.  Kernel A is the default, Kernel B takes
over below a stated count and runs everywhere in the ``NetCal-M`` arm, and the
disagreement between them is a reported deliverable: the equating section's advice
is to "treat disagreement between the two as a misspecification signal".

**The latent level, and what connects observers who never met.**  Kernel A
eliminates ``L_t``, so it says nothing about the absolute level; the level comes
from the anchor, where ``k_ref = 1`` exactly by construction (ADR-003).  For an
observer with no usable edge, nothing pairwise applies at all, and what identifies
it is the requirement that the activity level it implies be *continuous* with the
level implied by the observers around it -- a penalized spline in ``log L(t)``.
This is the satellite lesson in GN form: "A latent G(t) in a Bayesian model is the
radiative-transfer transfer standard, so exact overlaps become unnecessary wherever
the latent-process prior is credible across gaps."  The research notes attach the
matching warning, and it is carried into the results doc rather than buried:
"**Disconnected eras can each float independently unless a smoothness or physical
prior on G(t) ... ties them. That prior then becomes the dominant source of
information. It must be stated explicitly and tested in sensitivity runs.**"

**Drift.**  Uto (2023)'s Markov severity prior, with its measured hyperparameters:
``log T_{o,y} = log T_{o,y-1} + sigma_o z``, ``log sigma_o ~ N(mu_sigma, 1)`` and
``mu_sigma = -2`` recommended.  Uto measured RMSE gains in every simulated
condition, growing "as data per parameter shrank" -- which is the 17th century --
and measured the sensitivity this step must report: ``mu_sigma = -5`` over-shrinks
real change and ``mu_sigma = 0`` is unstable.  Careers are **not** cut at detected
breaks: "every cut ... raises effective resistance and lets real change leak away",
and a random segmentation already explains about half the break variance.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax.scipy.special import gammaln, logsumexp

jax.config.update("jax_enable_x64", True)

from gn_next_gen.graph.overlap import (  # noqa: E402
    OverlapGraph,
    PairTable,
    build_overlap_graph,
    decimal_year,
    pair_table,
)
from gn_next_gen.models.activity_map import ActivityMap, smooth_counts  # noqa: E402
from gn_next_gen.models.emissions import (  # noqa: E402
    MAX_COUNT,
    DetectionGrid,
    bilinear,
)

#: Smallest and largest acuity threshold the fit will consider, in MSH.  Wider than
#: the published grids (A157 caps at 300 msd = 150 MSH) so a pinned estimate is a
#: finding rather than an artifact of the bound -- Step 2 found both acuity methods
#: pinned at their 300 MSH ceiling for quiet-era observers.
THRESHOLD_LO_MSH = 1.0
THRESHOLD_HI_MSH = 500.0

#: Prior location for the per-observer drift scale, ``log sigma_o ~ N(mu, 1)``.
#: Uto (2023) recommends -2; -5 over-shrinks real change and 0 is unstable.
DRIFT_LOG_SIGMA_MEAN = -2.0
DRIFT_LOG_SIGMA_SD = 1.0

#: Prior on each observer's base threshold, in log MSH.  Centred near the Step 2
#: generator's population (log-mean 35 MSH) and deliberately loose.
THRESHOLD_LOG_MEAN = float(np.log(35.0))
THRESHOLD_LOG_SD = 1.0

#: Prior on the split rate, on a log scale.
SPLIT_LOG_MEAN = float(np.log(0.03))
SPLIT_LOG_SD = 1.0

#: Second-difference penalty on the activity spline, per knot.  The weight that
#: decides how much of an isolated observer's signal is attributed to the observer
#: and how much to the Sun, so it is a declared sensitivity, not a constant.
LAMBDA_PENALTY = 20.0

#: Below this career-mean count a pair is handed to Kernel B, where the
#: conditional-binomial reduction is documented to break down.
LOW_COUNT_THRESHOLD = 2.0


# ---------------------------------------------------------------------------
# Assembled inputs
# ---------------------------------------------------------------------------


@dataclass
class NetworkInputs:
    """Everything the likelihood reads, indexed and ready for JAX.

    The observer-year is the unit: one acuity per observer per calendar year, which
    is the resolution the drift prior works at and the resolution that makes the
    pair table collapsible.
    """

    stations: np.ndarray
    years: np.ndarray
    slot_station: np.ndarray
    slot_year: np.ndarray
    slot_of: dict[tuple[int, int], int]
    station_slot_start: np.ndarray
    station_n_slots: np.ndarray
    pair_slot_a: np.ndarray
    pair_slot_b: np.ndarray
    pair_count_a: np.ndarray
    pair_count_b: np.ndarray
    pair_weight: np.ndarray
    pair_rho: np.ndarray
    pair_low_count: np.ndarray
    observations: pd.DataFrame
    obs_slot: np.ndarray
    anchor_station: int
    anchor_slots: np.ndarray
    detection: DetectionGrid
    activity: np.ndarray
    documented_breaks: np.ndarray
    graph: OverlapGraph
    meta: dict = field(default_factory=dict)

    @property
    def n_stations(self) -> int:
        return len(self.stations)

    @property
    def n_slots(self) -> int:
        return len(self.slot_station)

    @property
    def n_pairs(self) -> int:
        return len(self.pair_slot_a)


def reference_autocorrelation(
    counts: np.ndarray, max_lag: int, *, detrend_days: int = 91
) -> np.ndarray:
    """Detrended autocorrelation of a daily count series, lags ``0..max_lag``.

    Detrending against a 91-day mean is what exposes the rotation structure rather
    than the solar cycle -- Step 1's f07 makes the point: the raw ACF "rides on the
    solar cycle", and only the detrended one shows the 13-day transit trough and
    the 27-day recurrence bump.  Here only the first few lags are used, since the
    day tolerance is small, but the same function serves a widened tolerance.
    """
    from gn_next_gen.stats.discrepancy import autocorrelation

    values = np.asarray(counts, dtype=float)
    acf = autocorrelation(values, max_lag=max(max_lag, 1), detrend_days=detrend_days)
    out = np.ones(max_lag + 1)
    usable = min(len(acf), max_lag + 1)
    out[:usable] = acf[:usable]
    out[0] = 1.0
    return np.clip(out, -0.999, 1.0)


def build_inputs(
    observations: pd.DataFrame,
    *,
    detection: DetectionGrid,
    activity_map: ActivityMap,
    anchor_station: int,
    graph: OverlapGraph | None = None,
    pairs: PairTable | None = None,
    metadata: pd.DataFrame | None = None,
    day_tolerance: int | None = None,
    min_overlap_days: int | None = None,
    min_distinct_values: int | None = None,
    reference_counts: pd.DataFrame | None = None,
) -> NetworkInputs:
    """Index the network into observer-year slots and matched pairs.

    Parameters
    ----------
    observations
        Long format ``date_ord, station, groups``, **including** the anchor as a
        station of its own.  :func:`anchor_observations` builds that row set from
        the reference catalogue.
    metadata
        Optional ``station, date_ord, note`` -- documented career breaks (ADR-004).
        Only documented breaks get a free jump; undocumented ones are left to the
        random walk, which is what the paired Step 2 case studies measure.
    """
    required = {"date_ord", "station", "groups"}
    missing = required - set(observations.columns)
    if missing:
        raise ValueError(f"observations is missing {sorted(missing)}")

    if graph is None:
        kwargs = {}
        if day_tolerance is not None:
            kwargs["day_tolerance"] = day_tolerance
        if min_overlap_days is not None:
            kwargs["min_overlap_days"] = min_overlap_days
        if min_distinct_values is not None:
            kwargs["min_distinct_values"] = min_distinct_values
        graph = build_overlap_graph(observations, **kwargs)

    # Keep only stations the graph kept, plus the anchor, so the parameter vector
    # has no entries nothing can constrain.
    keep = set(int(s) for s in graph.stations) | {int(anchor_station)}
    frame = observations[observations["station"].isin(keep)].copy()
    frame["year"] = np.floor(decimal_year(frame["date_ord"].to_numpy())).astype(np.int64)

    stations = np.sort(frame["station"].unique()).astype(np.int64)
    slots = (
        frame[["station", "year"]]
        .drop_duplicates()
        .sort_values(["station", "year"])
        .reset_index(drop=True)
    )
    slot_station = slots["station"].to_numpy(dtype=np.int64)
    slot_year = slots["year"].to_numpy(dtype=np.int64)
    slot_of = {
        (int(s), int(y)): i for i, (s, y) in enumerate(zip(slot_station, slot_year, strict=True))
    }
    starts = np.searchsorted(slot_station, stations, side="left")
    counts_per = np.searchsorted(slot_station, stations, side="right") - starts

    years = np.arange(int(slot_year.min()), int(slot_year.max()) + 1)

    # Activity level per calendar year, from the reference where it exists and from
    # the map elsewhere.  The outer loop overwrites this each pass.
    activity = np.full(len(years), np.nan)
    if reference_counts is not None:
        smoothed = smooth_counts(
            reference_counts["date_ord"].to_numpy(),
            reference_counts["n_groups"].to_numpy(),
        )
        by_year = (
            pd.DataFrame(
                {
                    "year": np.floor(decimal_year(smoothed.index.to_numpy())).astype(np.int64),
                    "gn": smoothed.to_numpy(),
                }
            )
            .groupby("year")["gn"]
            .mean()
        )
        aligned = by_year.reindex(years)
        activity = np.asarray(activity_map(aligned.to_numpy()), dtype=float)
    activity = np.where(np.isfinite(activity), activity, activity_map(np.array(1.0)))
    activity = np.clip(activity, detection.activity.min(), detection.activity.max())

    if pairs is None:
        pairs = pair_table(frame, graph, collapse=True)

    station_a, station_b = pairs.station_pairs()
    pair_station_a = station_a[pairs.edge]
    pair_station_b = station_b[pairs.edge]

    # Pack (station, year) into one sorted key and look both sides up with
    # searchsorted.  A dict lookup per row is 4.7M Python iterations on the real
    # network and was the single slowest thing in the assembly, at 15 s.
    year_base = int(slot_year.min())
    year_span = int(slot_year.max()) - year_base + 1

    def pack(station, year):
        return station.astype(np.int64) * year_span + (year.astype(np.int64) - year_base)

    slot_keys = pack(slot_station, slot_year)
    order = np.argsort(slot_keys, kind="stable")
    sorted_keys = slot_keys[order]

    def lookup(station, year):
        wanted = pack(np.asarray(station), np.asarray(year))
        position = np.searchsorted(sorted_keys, wanted)
        clamped = np.clip(position, 0, len(sorted_keys) - 1)
        found = sorted_keys[clamped] == wanted
        return order[clamped], found

    pair_slot_a, found_a = lookup(pair_station_a, pairs.year)
    if not found_a.all():
        raise AssertionError("observer a's slot must exist by construction")

    # Observer b's day can fall in the neighbouring year when the lag crosses a
    # year boundary.  Those rows fall back to b's own nearest year.
    pair_slot_b, found_b = lookup(pair_station_b, pairs.year)
    if not found_b.all():
        missing = np.flatnonzero(~found_b)
        station_first = {
            int(s): (int(starts[i]), int(counts_per[i])) for i, s in enumerate(stations)
        }
        for i in missing:
            start, length = station_first[int(pair_station_b[i])]
            available = slot_year[start : start + length]
            nearest = int(np.argmin(np.abs(available - int(pairs.year[i]))))
            pair_slot_b[i] = start + nearest

    acf = reference_autocorrelation(
        _anchor_daily(frame, int(anchor_station)), int(pairs.meta["tolerance"])
    )
    pair_rho = acf[np.abs(pairs.lag).astype(int)]

    career_mean = frame.groupby("station")["groups"].mean()
    low_a = career_mean.reindex(pair_station_a).to_numpy() < LOW_COUNT_THRESHOLD
    low_b = career_mean.reindex(pair_station_b).to_numpy() < LOW_COUNT_THRESHOLD
    pair_low_count = low_a | low_b

    obs_slot = np.array(
        [
            slot_of[(int(s), int(y))]
            for s, y in zip(frame["station"].to_numpy(), frame["year"].to_numpy(), strict=True)
        ],
        dtype=np.int64,
    )

    anchor_slots = np.flatnonzero(slot_station == int(anchor_station))

    documented = np.zeros(len(slot_station), dtype=bool)
    if metadata is not None and len(metadata):
        notes = metadata.copy()
        notes["year"] = np.floor(decimal_year(notes["date_ord"].to_numpy())).astype(np.int64)
        for station, year in zip(notes["station"], notes["year"], strict=True):
            key = (int(station), int(year))
            if key in slot_of:
                documented[slot_of[key]] = True

    return NetworkInputs(
        stations=stations,
        years=years,
        slot_station=slot_station,
        slot_year=slot_year,
        slot_of=slot_of,
        station_slot_start=starts,
        station_n_slots=counts_per,
        pair_slot_a=pair_slot_a,
        pair_slot_b=pair_slot_b,
        pair_count_a=pairs.count_a.astype(np.int64),
        pair_count_b=pairs.count_b.astype(np.int64),
        pair_weight=pairs.weight.astype(float),
        pair_rho=pair_rho,
        pair_low_count=pair_low_count,
        observations=frame,
        obs_slot=obs_slot,
        anchor_station=int(anchor_station),
        anchor_slots=anchor_slots,
        detection=detection,
        activity=activity,
        documented_breaks=documented,
        graph=graph,
        meta={
            "n_stations": len(stations),
            "n_slots": len(slot_station),
            "n_pair_rows": pairs.n_rows,
            "n_pair_observations": pairs.n_pairs,
            "pair_compression": pairs.meta.get("compression", 1.0),
            "tolerance": int(pairs.meta["tolerance"]),
            "low_count_rows": int(pair_low_count.sum()),
            "anchor_station": int(anchor_station),
            "anchor_slots": int(len(anchor_slots)),
            "acf": acf.tolist(),
            "year_lo": int(years.min()),
            "year_hi": int(years.max()),
        },
    )


def _anchor_daily(frame: pd.DataFrame, anchor_station: int) -> np.ndarray:
    """The anchor's own counts on a dense daily grid, for the ACF."""
    rows = frame[frame["station"] == anchor_station]
    if rows.empty:
        return np.zeros(1)
    days = rows["date_ord"].to_numpy(dtype=np.int64)
    lo, hi = int(days.min()), int(days.max())
    dense = np.full(hi - lo + 1, np.nan)
    dense[days - lo] = rows["groups"].to_numpy(dtype=float)
    return dense


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

#: Innovation scale allowed at a *documented* career break.  Metadata-documented
#: breaks get a free jump at this scale and no prior penalty; undocumented ones are
#: left to the random walk, which is what the paired Step 2 case studies
#: (``break-documented`` / ``break-undocumented``) exist to price -- documentation
#: was worth a factor of 2-3 to UEA21 and GRP.
BREAK_SCALE = 1.0


@dataclass(frozen=True)
class ParameterLayout:
    """Where each block of parameters sits in the flat vector L-BFGS sees."""

    n_stations: int
    n_slots: int
    anchor_row: int
    reference_threshold_msh: float

    @property
    def size(self) -> int:
        return 3 * self.n_stations + self.n_slots + 1

    def slices(self) -> dict[str, slice]:
        n, m = self.n_stations, self.n_slots
        return {
            "log_threshold": slice(0, n),
            "log_split": slice(n, 2 * n),
            "log_drift_sigma": slice(2 * n, 3 * n),
            "zeta": slice(3 * n, 3 * n + m),
            "log_kappa0": slice(3 * n + m, 3 * n + m + 1),
        }


def layout_for(inputs: NetworkInputs) -> ParameterLayout:
    anchor_row = int(np.flatnonzero(inputs.stations == inputs.anchor_station)[0])
    return ParameterLayout(
        n_stations=inputs.n_stations,
        n_slots=inputs.n_slots,
        anchor_row=anchor_row,
        reference_threshold_msh=float(inputs.detection.reference_threshold_msh),
    )


def initial_parameters(inputs: NetworkInputs, layout: ParameterLayout) -> np.ndarray:
    """A starting vector: everyone at the prior mean, no drift, no splitting.

    Deliberately **not** symmetric in a way that can vanish a gradient.  Step 1
    recorded the matching trap for the two-class filter: starting both classes
    equal put the optimizer at a point where the gradient along their difference
    was exactly 0.0, "so L-BFGS returns the start unchanged and the two-class fit
    silently collapses".  Here the anchor is pinned and everyone else starts away
    from it, so no such point is reachable.
    """
    theta = np.zeros(layout.size)
    s = layout.slices()
    theta[s["log_threshold"]] = THRESHOLD_LOG_MEAN
    theta[s["log_split"]] = SPLIT_LOG_MEAN
    theta[s["log_drift_sigma"]] = DRIFT_LOG_SIGMA_MEAN
    theta[s["zeta"]] = 0.0
    theta[s["log_kappa0"]] = np.log(5.0)
    return theta


def _unpack(theta, layout: ParameterLayout):
    s = layout.slices()
    log_threshold = theta[s["log_threshold"]]
    log_split = theta[s["log_split"]]
    log_drift_sigma = theta[s["log_drift_sigma"]]
    zeta = theta[s["zeta"]]
    log_kappa0 = theta[s["log_kappa0"]][0]

    # The anchor is pinned, not fitted: k_ref = 1 exactly, no drift, no splitting.
    # This is the one constraint that makes the absolute scale identified at all
    # (ADR-003); without it the whole system has an exact null mode, which is the
    # degeneracy SDSS calls "the zero point of the calibration ... exactly
    # degenerate".
    log_threshold = log_threshold.at[layout.anchor_row].set(jnp.log(layout.reference_threshold_msh))
    log_split = log_split.at[layout.anchor_row].set(-50.0)
    log_drift_sigma = log_drift_sigma.at[layout.anchor_row].set(-50.0)
    return log_threshold, log_split, log_drift_sigma, zeta, log_kappa0


def log_rate_ratio(theta, inputs: NetworkInputs, layout: ParameterLayout, static):
    """``log r_o(y) = log k_o(y) + log(1 + s_o)`` for every observer-year slot.

    ``r`` and not ``k`` because that is what a Bradley-Terry likelihood sees: a
    coarse observer and a splitting observer with the same mean are the same thing
    to Kernel A.  The split rate is identified only through the shape terms.
    """
    log_threshold, log_split, log_drift_sigma, zeta, _ = _unpack(theta, layout)

    sigma = jnp.where(
        static["documented"],
        BREAK_SCALE,
        jnp.exp(log_drift_sigma)[static["slot_station_row"]],
    )
    innovation = sigma * zeta
    # Cumulative sum within each station's run of years: one global cumsum, minus
    # the value at the run's start.  The slots are sorted by (station, year), which
    # build_inputs guarantees.
    cumulative = jnp.cumsum(innovation)
    start = cumulative[static["slot_run_start"]] - innovation[static["slot_run_start"]]
    drift = cumulative - start

    log_T = jnp.clip(
        log_threshold[static["slot_station_row"]] + drift,
        jnp.log(THRESHOLD_LO_MSH),
        jnp.log(THRESHOLD_HI_MSH),
    )
    log_k = bilinear(
        static["log_threshold_grid"],
        static["activity_grid"],
        static["log_k_table"],
        log_T,
        static["slot_activity"],
    )
    log_r = log_k + jnp.log1p(jnp.exp(log_split))[static["slot_station_row"]]
    return log_r, log_T


def _betaln(a, b):
    return gammaln(a) + gammaln(b) - gammaln(a + b)


def pairwise_loglik_conditional(log_r, inputs_static):
    """Kernel A: beta-binomial on each matched pair's own total.

    The latent daily level cancels exactly, so this term is informative about the
    *ratio* of two observers' rates and silent about the level -- which is why the
    anchor, and not this term, sets the scale.
    """
    y_a = inputs_static["count_a"]
    y_b = inputs_static["count_b"]
    n = y_a + y_b
    weight = inputs_static["weight"]
    rho = inputs_static["rho"]
    kappa0 = inputs_static["kappa0"]

    r_a = jnp.exp(log_r[inputs_static["slot_a"]])
    r_b = jnp.exp(log_r[inputs_static["slot_b"]])
    p = jnp.clip(r_a / (r_a + r_b), 1e-9, 1.0 - 1e-9)

    combinatorial = gammaln(n + 1.0) - gammaln(y_a + 1.0) - gammaln(y_b + 1.0)
    binomial = combinatorial + y_a * jnp.log(p) + y_b * jnp.log1p(-p)

    # Overdispersion grows as the lag decorrelates the two observers' Suns.
    spread = jnp.clip(1.0 - rho, 0.0, 1.0)
    kappa = kappa0 / jnp.maximum(spread, 1e-12)
    alpha = p * kappa
    beta = (1.0 - p) * kappa
    overdispersed = combinatorial + _betaln(y_a + alpha, y_b + beta) - _betaln(alpha, beta)

    # dt = 0 is a plain binomial; a lag is honestly overdispersed.  Both branches
    # are evaluated and selected, rather than pushing kappa to infinity, because
    # differencing two very large gammaln values loses the precision the float64
    # accumulation exists to protect.
    term = jnp.where(spread <= 1e-12, binomial, overdispersed)
    term = jnp.where(n > 0, term, 0.0)
    resample = inputs_static["year_weight"][inputs_static["pair_year"]]
    return jnp.sum(weight * resample * term)


def pairwise_loglik_marginal(log_r, inputs_static, *, rows=None):
    """Kernel B: the exact marginalization over the latent count.

    ``P(y_b | y_a) = sum_N P(y_b | N, b) P(N | y_a, a)`` with ``P(N | y_a)``
    proportional to ``P(y_a | N, a)`` times the latent prior the outer loop
    supplies from the current activity estimate.

    The difference from Kernel A is not cosmetic and is the reason both exist.
    Kernel A *eliminates* the latent level by conditioning on the pair's total, so
    it is exactly right about the ratio and silent about everything else.  Kernel B
    *integrates* the latent level against a prior, so it uses the absolute counts
    as well -- which is what makes it work where zeros dominate and the
    conditional-binomial reduction is documented to break down, and what makes it
    sensitive to a misspecified prior in a way Kernel A is not.  Their disagreement
    is therefore informative about misspecification rather than about arithmetic.

    ``rows`` restricts the term to a subset, which the hybrid kernel uses.  It is a
    real restriction and not a weight of zero: only two columns of each
    ``(K, K)`` emission table are ever read, but materializing the tables for
    every row anyway would be a ``4.7M x 42 x 42`` tensor -- 66 GB on a real
    benchmark world.  :func:`_emission_column` gathers the two columns directly,
    and ``rows`` keeps even that to the pairs that need it.

    One stated approximation: the emission library is indexed by the rate ratio
    ``r`` with the split rate folded in, rather than on a two-dimensional
    ``(k, s)`` grid.  In the low-count regime this kernel serves, ``N`` is 0-3 and
    the thin-versus-split distinction moves ``P(y | N)`` by less than the sampling
    noise; the shape information about ``s`` is carried by the occupancy term.
    """
    log_emission = inputs_static["log_emission"]
    log_prior_n = inputs_static["log_prior_n"]
    k_grid_log = inputs_static["k_grid_log"]

    slot_a = inputs_static["slot_a"]
    slot_b = inputs_static["slot_b"]
    count_a = inputs_static["count_a"]
    count_b = inputs_static["count_b"]
    weight = inputs_static["weight"]
    resample = inputs_static["year_weight"][inputs_static["pair_year"]]
    if rows is not None:
        slot_a = slot_a[rows]
        slot_b = slot_b[rows]
        count_a = count_a[rows]
        count_b = count_b[rows]
        weight = weight[rows]
        resample = resample[rows]

    column_a = _emission_column(log_emission, k_grid_log, log_r[slot_a], count_a)
    column_b = _emission_column(log_emission, k_grid_log, log_r[slot_b], count_b)

    log_post = log_prior_n[slot_a] + column_a
    log_post = log_post - logsumexp(log_post, axis=1, keepdims=True)
    log_term = logsumexp(log_post + column_b, axis=1)
    return jnp.sum(weight * resample * log_term)


def _emission_column(log_emission, k_grid_log, log_r, counts):
    """``log P(y = counts[i] | N, r_i)`` for every ``N``, one row per pair.

    Gathers the two needed columns straight out of the library instead of
    building each row's whole ``(K, K)`` table and then slicing it.  The slice was
    always what got used; the table was a 66 GB intermediate on a real benchmark
    world, and it is the reason the first hybrid fit never finished.
    """
    position = jnp.clip(jnp.searchsorted(k_grid_log, log_r) - 1, 0, len(k_grid_log) - 2)
    lo = k_grid_log[position]
    hi = k_grid_log[position + 1]
    t = jnp.clip((log_r - lo) / (hi - lo), 0.0, 1.0)[:, None]
    index = counts.astype(jnp.int32)
    low = log_emission[position, :, index]
    high = log_emission[position + 1, :, index]
    return low * (1.0 - t) + high * t


def occupancy_loglik(log_r, inputs_static):
    """The fingerprint term, written as a likelihood rather than a grid search.

    For each observer-year, the number of spotless reports against what an observer
    of that rate would have produced, with the latent count distribution taken from
    the reference **at that activity level**::

        P(y = 0 | r, AL) = sum_N pi_N(AL) P(0 | N, r)

    which is the occupancy form the spec names, ``1 - (1 - r)^N`` generalized to an
    area-dependent detection function -- the Royle-Nichols curve
    ``ecology_imperfect_detection.md`` identifies as "the formal version of the ADF
    fingerprint".  No EMD, no ``NBEST`` grid search (``reuse_map.md``'s Drop list).

    **How much this term can possibly be worth, measured.**  Conditioned on
    activity level, the reference's own daily count distribution is almost exactly
    Poisson: variance-to-mean runs **0.88 to 1.17** across twelve activity bins,
    mostly within 5% of 1.  Thinning a Poisson gives a Poisson, so the shape of a
    record's count distribution carries *almost no information* about the thinning
    rate once activity is conditioned on.  The term is kept, at weight 1 (PI,
    2026-11-24), and the ``w``-ablation measures what it actually contributes.

    This is also the sharpest available explanation of Step 2's f23.  UEA21 and GRP
    appear to extract acuity from exactly this statistic -- but they do not
    condition on activity, so what they are really measuring is the activity
    difference between the observer's era and the anchor epoch.  That is not a
    confound sitting beside their signal; on this evidence it **is** their signal,
    which is why their recovered acuity tracks era activity at rho = -0.62 and
    -0.73 and pins at the grid ceiling for quiet-era observers.
    """
    log_emission = inputs_static["log_emission"]
    k_grid_log = inputs_static["k_grid_log"]
    pmf = inputs_static["slot_count_pmf"]

    zero_column = _emission_column(log_emission, k_grid_log, log_r, jnp.zeros(log_r.shape[0]))
    p_zero = jnp.clip(jnp.sum(pmf * jnp.exp(zero_column), axis=1), 1e-9, 1.0 - 1e-9)
    resample = inputs_static["year_weight"][inputs_static["slot_year_index"]]
    n_days = inputs_static["slot_n_days"] * resample
    n_zero = inputs_static["slot_n_zero"] * resample
    term = n_zero * jnp.log(p_zero) + (n_days - n_zero) * jnp.log1p(-p_zero)
    return jnp.sum(jnp.where(n_days > 0, term, 0.0))


def activity_spline_penalty(log_r, inputs_static):
    """What ties an observer with no usable overlap to the rest of the record.

    Each year's activity level is profiled from every observer who reported in it,
    ``L_y = mean(y_i / r_i)``, and the penalty is the squared second difference of
    ``log L`` across *consecutive* years.  An isolated observer whose rate is wrong
    makes its own years step away from its neighbours', and the penalty is what
    notices.

    This is the satellite lesson in GN form -- "a latent G(t) ... is the
    radiative-transfer transfer standard, so exact overlaps become unnecessary
    wherever the latent-process prior is credible across gaps" -- and it carries
    the matching warning, which the results doc states rather than buries:
    "**Disconnected eras can each float independently unless a smoothness or
    physical prior on G(t) ... ties them. That prior then becomes the dominant
    source of information.**"  Its weight is therefore a declared sensitivity.

    Only triples of consecutive calendar years contribute: a second difference
    taken across a fifty-year gap in the record is not a statement about smoothness.
    """
    calibrated = inputs_static["obs_counts"] / jnp.exp(log_r[inputs_static["obs_slot"]])
    total = jnp.zeros(inputs_static["n_years"]).at[inputs_static["obs_year"]].add(calibrated)
    n = inputs_static["year_n_obs"]
    level = total / jnp.maximum(n, 1.0)
    log_level = jnp.log(jnp.maximum(level, 1e-6))
    triples = inputs_static["year_triples"]
    second = log_level[triples[:, 0]] - 2.0 * log_level[triples[:, 1]] + log_level[triples[:, 2]]
    return -inputs_static["lambda_penalty"] * jnp.sum(second**2)


def log_prior(theta, layout: ParameterLayout, static):
    """Hierarchical priors: thresholds, split rates, and the Uto drift walk."""
    log_threshold, log_split, log_drift_sigma, zeta, log_kappa0 = _unpack(theta, layout)
    free = static["free_station"]

    threshold_term = jnp.sum(
        jnp.where(free, -0.5 * ((log_threshold - THRESHOLD_LOG_MEAN) / THRESHOLD_LOG_SD) ** 2, 0.0)
    )
    split_term = jnp.sum(
        jnp.where(free, -0.5 * ((log_split - SPLIT_LOG_MEAN) / SPLIT_LOG_SD) ** 2, 0.0)
    )
    # log sigma_o ~ N(mu, 1): Uto (2023)'s recommended location is mu = -2.  The
    # location is what decides how much apparent change is attributed to the
    # observer rather than to the Sun, so it is a reported sensitivity.
    drift_term = jnp.sum(
        jnp.where(
            free,
            -0.5 * ((log_drift_sigma - static["drift_log_sigma_mean"]) / DRIFT_LOG_SIGMA_SD) ** 2,
            0.0,
        )
    )
    # Non-centred innovations are standard normal -- except at a documented break,
    # where the jump is free and carries no penalty.
    zeta_term = -0.5 * jnp.sum(jnp.where(static["documented"], 0.0, zeta**2))
    kappa_term = -0.5 * ((log_kappa0 - jnp.log(5.0)) / 2.0) ** 2
    return threshold_term + split_term + drift_term + zeta_term + kappa_term


def make_static(
    inputs: NetworkInputs,
    *,
    kernel: str = "conditional",
    fingerprint_weight: float = 1.0,
    lambda_penalty: float = LAMBDA_PENALTY,
    drift_log_sigma_mean: float = DRIFT_LOG_SIGMA_MEAN,
    max_count: int = MAX_COUNT,
    count_pmf: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict:
    """Freeze every array the objective reads into JAX form, once."""
    from gn_next_gen.models.emissions import empirical_count_pmf, k_grid, log_emission_library

    station_row = {int(s): i for i, s in enumerate(inputs.stations)}
    slot_station_row = np.array([station_row[int(s)] for s in inputs.slot_station], dtype=np.int64)
    run_start = np.zeros(inputs.n_slots, dtype=np.int64)
    for i, start in enumerate(inputs.station_slot_start):
        run_start[start : start + inputs.station_n_slots[i]] = start

    year_index = {int(y): i for i, y in enumerate(inputs.years)}
    slot_year_index = np.array([year_index[int(y)] for y in inputs.slot_year], dtype=np.int64)
    slot_activity = inputs.activity[slot_year_index]
    pair_year_index = slot_year_index[inputs.pair_slot_a]

    observations = inputs.observations
    obs_year = np.array(
        [year_index[int(y)] for y in observations["year"].to_numpy()], dtype=np.int64
    )
    year_n_obs = np.bincount(obs_year, minlength=len(inputs.years)).astype(float)

    present = np.flatnonzero(year_n_obs > 0)
    present_set = set(present.tolist())
    triples = np.array(
        [(i - 1, i, i + 1) for i in present if (i - 1) in present_set and (i + 1) in present_set],
        dtype=np.int64,
    ).reshape(-1, 3)

    # Spotless reports per observer-year, for the occupancy term.
    grouped = observations.assign(slot=inputs.obs_slot).groupby("slot")["groups"]
    slot_n_days = np.zeros(inputs.n_slots)
    slot_n_zero = np.zeros(inputs.n_slots)
    slot_n_days[grouped.size().index.to_numpy()] = grouped.size().to_numpy()
    zeros = (
        observations.assign(slot=inputs.obs_slot)
        .groupby("slot")["groups"]
        .apply(lambda block: int((block == 0).sum()))
    )
    slot_n_zero[zeros.index.to_numpy()] = zeros.to_numpy()

    if count_pmf is not None:
        pmf, pmf_activity = count_pmf
    else:
        pmf, pmf_activity = empirical_count_pmf(
            inputs.detection.survey,
            reference_threshold_msh=inputs.detection.reference_threshold_msh,
            max_count=max_count,
        )
    bin_of = np.argmin(np.abs(slot_activity[:, None] - pmf_activity[None, :]), axis=1)
    slot_count_pmf = pmf[bin_of]

    log_emission, k_values, _splits = log_emission_library(
        k_values=k_grid(), split_values=np.array([0.0]), max_count=max_count
    )
    log_emission = log_emission[:, 0]

    # Latent-count prior per slot, from the reference's own activity-conditional
    # distribution rather than an assumed Poisson.
    log_prior_n = np.log(np.maximum(slot_count_pmf, 1e-300))

    free_station = inputs.stations != inputs.anchor_station

    static = {
        "slot_station_row": jnp.asarray(slot_station_row),
        "slot_run_start": jnp.asarray(run_start),
        "slot_activity": jnp.asarray(slot_activity),
        "documented": jnp.asarray(inputs.documented_breaks),
        "free_station": jnp.asarray(free_station),
        "log_threshold_grid": jnp.asarray(inputs.detection.log_threshold),
        "activity_grid": jnp.asarray(inputs.detection.activity),
        "log_k_table": jnp.asarray(inputs.detection.log_k),
        "slot_a": jnp.asarray(inputs.pair_slot_a),
        "slot_b": jnp.asarray(inputs.pair_slot_b),
        "count_a": jnp.asarray(inputs.pair_count_a.astype(float)),
        "count_b": jnp.asarray(inputs.pair_count_b.astype(float)),
        "weight": jnp.asarray(inputs.pair_weight),
        "rho": jnp.asarray(inputs.pair_rho),
        "low_count": jnp.asarray(inputs.pair_low_count),
        "low_count_rows": jnp.asarray(np.flatnonzero(inputs.pair_low_count)),
        "obs_counts": jnp.asarray(observations["groups"].to_numpy(dtype=float)),
        "obs_slot": jnp.asarray(inputs.obs_slot),
        "obs_year": jnp.asarray(obs_year),
        "year_n_obs": jnp.asarray(year_n_obs),
        "year_triples": jnp.asarray(triples),
        "n_years": len(inputs.years),
        "slot_n_days": jnp.asarray(slot_n_days),
        "slot_n_zero": jnp.asarray(slot_n_zero),
        "slot_count_pmf": jnp.asarray(slot_count_pmf),
        "log_prior_n": jnp.asarray(log_prior_n),
        "log_emission": jnp.asarray(log_emission),
        "k_grid_log": jnp.asarray(np.log(k_values)),
        "pair_year": jnp.asarray(pair_year_index),
        "slot_year_index": jnp.asarray(slot_year_index),
        "year_weight": jnp.ones(len(inputs.years)),
        "lambda_penalty": float(lambda_penalty),
        "fingerprint_weight": float(fingerprint_weight),
        "drift_log_sigma_mean": float(drift_log_sigma_mean),
        "kernel": kernel,
    }
    return static


def composite_loglik(theta, inputs: NetworkInputs, layout: ParameterLayout, static) -> jax.Array:
    """The whole objective: pairwise + occupancy + activity smoothness + priors.

    One sum, at weight 1 for the occupancy term (PI decision 4).  The day-block
    bootstrap resamples every term's score jointly, so the information each really
    contributes -- including the part they share on the same observer-days -- is
    *measured* by the Godambe correction rather than guessed at by a weight.
    """
    _, _, _, _, log_kappa0 = _unpack(theta, layout)
    log_r, _log_T = log_rate_ratio(theta, inputs, layout, static)
    pair_static = dict(static)
    pair_static["kappa0"] = jnp.exp(log_kappa0)

    if static["kernel"] == "conditional":
        pairwise = pairwise_loglik_conditional(log_r, pair_static)
    elif static["kernel"] == "marginal":
        pairwise = pairwise_loglik_marginal(log_r, pair_static)
    elif static["kernel"] == "hybrid":
        # Kernel A on every pair with the low-count rows zero-weighted (cheap: the
        # conditional kernel is a few scalars per row), and Kernel B on the
        # low-count rows *only*, as a real row subset -- see
        # ``pairwise_loglik_marginal`` on why a zero weight is not enough there.
        low = static["low_count"]
        high_static = dict(pair_static)
        high_static["weight"] = jnp.where(low, 0.0, pair_static["weight"])
        pairwise = pairwise_loglik_conditional(log_r, high_static)
        rows = static["low_count_rows"]
        if rows is not None and len(rows):
            pairwise = pairwise + pairwise_loglik_marginal(log_r, pair_static, rows=rows)
    else:
        raise ValueError(
            f"unknown kernel {static['kernel']!r}; expected 'conditional', 'marginal' or 'hybrid'"
        )

    occupancy = static["fingerprint_weight"] * occupancy_loglik(log_r, static)
    smoothness = activity_spline_penalty(log_r, static)
    return pairwise + occupancy + smoothness + log_prior(theta, layout, static)


# ---------------------------------------------------------------------------
# The anchor as an observer, and the real network
# ---------------------------------------------------------------------------

#: Station id the anchor survey joins the observer network under.  Matches
#: ``benchmark.scenarios.REFERENCE_STATION`` so the benchmark and the real record
#: take the same code path, and far above v1.3's 1-776 so it cannot collide.
ANCHOR_STATION = 9001

#: The ADR-009 anchorings, by cycle number rather than by date, so they "survive
#: small boundary movements in a future upstream refit".  ``rgo_12_19`` is the
#: named extension ADR-009 records as "a named extension to test in Step 3",
#: excluded by default on D5 grounds and run here as a sensitivity only.
ANCHORINGS = {
    "debrecen_21_23": {"survey": "debrecen", "cycles": (21, 22, 23)},
    "rgo_15_19": {"survey": "rgo", "cycles": (15, 16, 17, 18, 19)},
    "rgo_12_19": {"survey": "rgo", "cycles": (12, 13, 14, 15, 16, 17, 18, 19)},
}


def anchor_span(anchoring: str) -> tuple[int, int]:
    """Day-ordinal span of an anchoring, from the frozen cycle-coverage table."""
    if anchoring not in ANCHORINGS:
        raise ValueError(f"unknown anchoring {anchoring!r}; expected {sorted(ANCHORINGS)}")
    from gn_next_gen.paths import REFERENCE_CYCLE_COVERAGE

    spec = ANCHORINGS[anchoring]
    table = pd.read_csv(REFERENCE_CYCLE_COVERAGE)
    rows = table[(table["survey"] == spec["survey"]) & (table["cycle"].isin(spec["cycles"]))]
    if len(rows) != len(spec["cycles"]):
        raise ValueError(f"{anchoring}: cycle coverage table is missing cycles")
    incomplete = rows.loc[~rows["complete"].astype(bool), "cycle"].tolist()
    if incomplete:
        raise ValueError(
            f"{anchoring}: cycles {incomplete} are not complete in {spec['survey']}; "
            "ADR-009 defines an anchor as whole cycles entirely covered by one survey"
        )
    return int(rows["first_ord"].min()), int(rows["last_ord"].max())


def anchor_observations(
    anchoring: str,
    *,
    threshold_msh: float | None = None,
    station: int = ANCHOR_STATION,
) -> tuple[pd.DataFrame, float]:
    """The anchor survey's own daily counts, as one more station in the network.

    ADR-003 is explicit that the reference is not truth: "other reference segments
    ... are treated as observers with their own parameters", and the survey
    literature says the same thing twice over -- Pan-STARRS found that "the
    reference survey, SDSS, contributed about as much to their mutual differences
    as Pan-STARRS itself", and the derivation concludes "RGO/Debrecen should be
    modeled as an observer with its own uncertainty, not as truth".

    What makes *this* segment special is only that its detection is **fixed**: it
    is the one observer with ``k = 1`` by definition, which is what pins the
    otherwise exactly degenerate absolute scale.

    Returns the observation rows and the threshold that defines the estimand.
    """
    from gn_next_gen.io.reference import daily_counts
    from gn_next_gen.reference.area_law import load_upstream_parameters

    spec = ANCHORINGS[anchoring]
    survey = spec["survey"]
    if threshold_msh is None:
        threshold_msh = float(load_upstream_parameters(survey).threshold_msh)
    first, last = anchor_span(anchoring)

    counts = daily_counts(float(threshold_msh), [survey])
    inside = counts[(counts["date_ord"] >= first) & (counts["date_ord"] <= last)]
    frame = pd.DataFrame(
        {
            "date_ord": inside["date_ord"].to_numpy(dtype=np.int64),
            "station": np.full(len(inside), int(station), dtype=np.int64),
            "groups": inside["n_groups"].to_numpy(dtype=np.int64),
            "source": "anchor",
        }
    )
    return frame, float(threshold_msh)


# ---------------------------------------------------------------------------
# From parameters to a series
# ---------------------------------------------------------------------------


def observer_table(
    theta: np.ndarray,
    inputs: NetworkInputs,
    layout: ParameterLayout,
    static,
    *,
    covariance: np.ndarray | None = None,
    cov_indices: np.ndarray | None = None,
    diagnostics: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per observer: the calibration factor, its acuity, its diagnostics.

    ``factor`` is ``1 / r_o``, the multiplier that turns this observer's count into
    a reference-equivalent one, averaged over its career; ``recovered_threshold_msh``
    is the fitted acuity.  The column names match
    :class:`gn_next_gen.baselines.base.CalibrationResult` so the Step 2 scorer reads
    this without changes.
    """
    log_r, log_T = log_rate_ratio(jnp.asarray(theta), inputs, layout, static)
    log_r = np.asarray(log_r)
    threshold = np.exp(np.asarray(log_T))

    frame = pd.DataFrame(
        {
            "station": inputs.slot_station,
            "year": inputs.slot_year,
            "log_r": log_r,
            "threshold_msh": threshold,
            "n_days": np.asarray(static["slot_n_days"]),
        }
    )
    weighted = frame.assign(w=frame["n_days"].clip(lower=1.0))
    per_station = weighted.groupby("station").apply(
        lambda block: pd.Series(
            {
                "factor": float(np.average(np.exp(-block["log_r"]), weights=block["w"])),
                "recovered_threshold_msh": float(
                    np.average(block["threshold_msh"], weights=block["w"])
                ),
                "drift_log_range": float(block["log_r"].max() - block["log_r"].min()),
                "n_years": int(len(block)),
            }
        ),
        include_groups=False,
    )
    per_station = per_station.reset_index()

    sd = _factor_sd(theta, inputs, layout, static, covariance, indices=cov_indices)
    per_station["factor_sd"] = per_station["station"].map(sd).to_numpy()
    per_station["reached"] = True
    per_station["n_hops"] = 0
    if diagnostics is not None:
        per_station = per_station.merge(diagnostics, on="station", how="left")
        per_station["reached"] = per_station["anchored"].fillna(False)
        per_station["n_hops"] = np.where(per_station["reached"], 1, -1)
        per_station.loc[per_station["station"] == inputs.anchor_station, "n_hops"] = 0
    return per_station


def _factor_sd(theta, inputs, layout, static, covariance, indices=None) -> pd.Series:
    """Per-observer sd of the calibration factor, by the delta method.

    ``sd(1/r) = sd(log r) / r``.  With no covariance supplied the column is NaN,
    which the scorer reads as "this method reports no uncertainty" -- a finding in
    its own right rather than a zero.

    The Jacobian is taken **only over the parameters that carry covariance**.
    Differentiating over the whole vector means one forward pass per parameter --
    5,688 of them on a real benchmark world against 699 that are not identically
    zero -- and it was the dominant cost of a fit before this.
    """
    stations = np.unique(inputs.slot_station)
    if covariance is None:
        return pd.Series(np.nan, index=stations)

    if indices is None:
        indices = np.flatnonzero(np.diag(covariance) > 0)
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) == 0:
        return pd.Series(np.nan, index=stations)

    full = jnp.asarray(theta)

    def log_r_of(sub):
        out, _ = log_rate_ratio(full.at[indices].set(sub), inputs, layout, static)
        return out

    jacobian = np.asarray(jax.jacfwd(log_r_of)(jnp.asarray(theta[indices])))
    block = covariance[np.ix_(indices, indices)]
    variance = np.einsum("ij,jk,ik->i", jacobian, block, jacobian)
    frame = pd.DataFrame(
        {
            "station": inputs.slot_station,
            "sd_log_r": np.sqrt(np.maximum(variance, 0.0)),
            "w": np.asarray(static["slot_n_days"]).clip(min=1.0),
        }
    )
    pooled = frame.groupby("station").apply(
        lambda block: float(np.average(block["sd_log_r"], weights=block["w"])),
        include_groups=False,
    )
    return pooled


def combine_series(
    theta: np.ndarray,
    inputs: NetworkInputs,
    layout: ParameterLayout,
    static,
    *,
    draws: np.ndarray | None = None,
) -> pd.DataFrame:
    """Daily reference-equivalent counts, with the uncertainty split Step 2 requires.

    ``gn_sd_factor`` is the part that comes from the observers' calibration factors
    -- identical on every day of a career, so it does **not** average down within a
    year -- and ``gn_sd_spread`` is the part from observers disagreeing on the day
    itself, which does.  Pooling the two as though both averaged down shrinks a
    yearly interval by up to ``sqrt(365) ~ 19``, which is the defect that made Step
    2's first coverage table report the opposite of what it now reports (§12.7).
    """
    observations = inputs.observations
    log_r, _ = log_rate_ratio(jnp.asarray(theta), inputs, layout, static)
    calibrated = observations["groups"].to_numpy(dtype=float) / np.exp(
        np.asarray(log_r)[inputs.obs_slot]
    )

    frame = pd.DataFrame(
        {
            "date_ord": observations["date_ord"].to_numpy(dtype=np.int64),
            "value": calibrated,
        }
    )
    grouped = frame.groupby("date_ord")["value"]
    mean = grouped.mean()
    spread = grouped.std(ddof=1).fillna(0.0)
    n_observers = grouped.size()

    if draws is None:
        factor_sd = pd.Series(np.nan, index=mean.index)
    else:
        # One ``bincount`` per draw instead of one pandas ``groupby`` per draw.  On
        # a real world that is ~10 ms against ~500 ms, i.e. seconds rather than
        # minutes for a 400-draw ensemble.
        days = mean.index.to_numpy()
        codes = np.searchsorted(days, frame["date_ord"].to_numpy())
        per_day = np.bincount(codes, minlength=len(days)).astype(float)
        counts = observations["groups"].to_numpy(dtype=float)
        per_draw = np.empty((len(draws), len(days)))
        for d, vector in enumerate(draws):
            log_r_d, _ = log_rate_ratio(jnp.asarray(vector), inputs, layout, static)
            values = counts / np.exp(np.asarray(log_r_d)[inputs.obs_slot])
            per_draw[d] = np.bincount(codes, weights=values, minlength=len(days)) / np.maximum(
                per_day, 1.0
            )
        factor_sd = pd.Series(per_draw.std(axis=0, ddof=1), index=mean.index)

    day_spread = spread / np.sqrt(np.maximum(n_observers.to_numpy(), 1))
    combined = np.sqrt(
        np.nan_to_num(factor_sd.to_numpy(), nan=0.0) ** 2 + day_spread.to_numpy() ** 2
    )
    if draws is None:
        combined = np.where(n_observers.to_numpy() > 1, day_spread.to_numpy(), np.nan)

    return pd.DataFrame(
        {
            "date_ord": mean.index.to_numpy(),
            "gn": mean.to_numpy(),
            "gn_sd": combined,
            "gn_sd_factor": np.nan_to_num(factor_sd.to_numpy(), nan=0.0),
            "gn_sd_spread": day_spread.to_numpy(),
            "n_observers": n_observers.to_numpy(),
        }
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# The driver
# ---------------------------------------------------------------------------


@dataclass
class NetworkFit:
    """Everything one calibration run produced."""

    theta: np.ndarray
    inputs: NetworkInputs
    layout: ParameterLayout
    static: dict
    series: pd.DataFrame
    observers: pd.DataFrame
    correction: object | None
    draws: np.ndarray | None
    passes: list[dict]
    meta: dict = field(default_factory=dict)


def _year_block_score(objective, theta, indices, static_builder):
    """A closure that reweights whole calendar years and returns the score.

    One gradient evaluation per bootstrap resample, no refitting -- which is what
    keeps 200 resamples affordable on the full network.

    **The year weights are a traced argument, not a closure variable.**  Building a
    fresh closure per resample makes JAX retrace and recompile the whole objective
    every time, which turns 200 cheap gradients into 200 compilations.  This is the
    same trap Step 1 recorded for ``TruncatedLikelihood``: hoist everything that
    does not change out of the inner loop, and the cost collapses.
    """
    full = jnp.asarray(theta)
    sub_theta = jnp.asarray(theta[indices])

    @jax.jit
    def gradient(weights):
        def reduced(sub):
            return objective(full.at[indices].set(sub), static_builder(weights))

        return jax.grad(reduced)(sub_theta)

    def score(multiplicity: np.ndarray) -> np.ndarray:
        return np.asarray(gradient(jnp.asarray(multiplicity, dtype=float)), dtype=float)

    return score


def calibrate_network(
    observations: pd.DataFrame,
    *,
    detection: DetectionGrid,
    activity_map: ActivityMap,
    anchor_station: int,
    name: str = "NetCal",
    kernel: str = "hybrid",
    metadata: pd.DataFrame | None = None,
    graph: OverlapGraph | None = None,
    catalogue: object | None = None,
    fingerprint_weight: float = 1.0,
    lambda_penalty: float = LAMBDA_PENALTY,
    drift_log_sigma_mean: float = DRIFT_LOG_SIGMA_MEAN,
    n_outer: int = 4,
    n_bootstrap: int = 200,
    n_draws: int = 400,
    seed: int = 0,
    maxiter: int = 600,
    correct: bool = True,
    day_tolerance: int | None = None,
    min_overlap_days: int | None = None,
    min_distinct_values: int | None = None,
) -> NetworkFit:
    """Fit every observer concurrently, then correct the uncertainty.

    The outer loop is the activity fixed point: the detection ratio needs an
    activity level, the activity level comes from the series, and the series needs
    the detection ratio.  Three to five passes settle it (``n_outer``).  It is a
    fixed point, not a circle -- the same structure Step 4 formalizes with a latent
    activity state.
    """
    from gn_next_gen.graph import diagnostics as graph_diagnostics
    from gn_next_gen.inference.composite import (
        draw_parameters,
        expand_covariance,
        fit_mode,
        godambe_correction,
    )

    anchor_rows = observations[observations["station"] == anchor_station]
    if anchor_rows.empty:
        raise ValueError(
            f"the anchor station {anchor_station} is absent from the observations; "
            "a network fit has no scale without it (ADR-003)"
        )
    reference_counts = pd.DataFrame(
        {
            "date_ord": anchor_rows["date_ord"].to_numpy(),
            "n_groups": anchor_rows["groups"].to_numpy(),
        }
    )

    count_pmf = None
    if catalogue is not None:
        from gn_next_gen.models.emissions import count_pmf_from_counts

        count_pmf = count_pmf_from_counts(reference_counts, catalogue.activity)

    inputs = build_inputs(
        observations,
        detection=detection,
        activity_map=activity_map,
        anchor_station=anchor_station,
        graph=graph,
        metadata=metadata,
        day_tolerance=day_tolerance,
        min_overlap_days=min_overlap_days,
        min_distinct_values=min_distinct_values,
        reference_counts=reference_counts,
    )
    layout = layout_for(inputs)
    theta = initial_parameters(inputs, layout)

    passes: list[dict] = []
    static = None
    stage_seconds: dict[str, float] = {}
    stage_start = time.time()
    for outer in range(max(int(n_outer), 1)):
        static = make_static(
            inputs,
            kernel=kernel,
            fingerprint_weight=fingerprint_weight,
            lambda_penalty=lambda_penalty,
            drift_log_sigma_mean=drift_log_sigma_mean,
            count_pmf=count_pmf,
        )

        def objective(vector, static=static, inputs=inputs, layout=layout):
            return composite_loglik(vector, inputs, layout, static)

        fit = fit_mode(objective, theta, maxiter=maxiter)
        theta = fit.theta

        series = combine_series(theta, inputs, layout, static)
        previous = inputs.activity.copy()
        inputs = _refresh_activity(inputs, series, activity_map, detection)
        shift = float(
            np.nanmax(
                np.abs(
                    np.log(np.maximum(inputs.activity, 1e-9)) - np.log(np.maximum(previous, 1e-9))
                )
            )
        )
        passes.append(
            {
                "pass": outer,
                "objective": fit.objective,
                "n_iterations": fit.n_iterations,
                "success": fit.success,
                "message": fit.message,
                "seconds": fit.seconds,
                "max_log_activity_shift": shift,
            }
        )
        if shift < 0.01:
            break

    stage_seconds["mode"] = time.time() - stage_start
    stage_start = time.time()

    static = make_static(
        inputs,
        kernel=kernel,
        fingerprint_weight=fingerprint_weight,
        lambda_penalty=lambda_penalty,
        drift_log_sigma_mean=drift_log_sigma_mean,
        count_pmf=count_pmf,
    )

    def final_objective(vector, static=static):
        return composite_loglik(vector, inputs, layout, static)

    correction = None
    draws = None
    covariance = None
    if correct:
        slices = layout.slices()
        # The anchor's own entries are pinned inside ``_unpack``, so they carry
        # exactly zero curvature.  Including them would put two certainly-null
        # directions into the covariance block and make every inflation ratio NaN.
        thresholds = np.delete(
            np.arange(slices["log_threshold"].start, slices["log_threshold"].stop),
            layout.anchor_row,
        )
        splits = np.delete(
            np.arange(slices["log_split"].start, slices["log_split"].stop),
            layout.anchor_row,
        )
        indices = np.concatenate(
            [
                thresholds,
                splits,
                np.arange(slices["log_kappa0"].start, slices["log_kappa0"].stop),
            ]
        )

        def static_builder(multiplicity):
            rebuilt = dict(static)
            rebuilt["year_weight"] = multiplicity
            return rebuilt

        def objective_with(vector, built):
            return composite_loglik(vector, inputs, layout, built)

        def deterministic(vector, static=static, inputs=inputs, layout=layout):
            """Priors plus smoothness: everything the bootstrap holds fixed."""
            return log_prior(vector, layout, static) + activity_spline_penalty(
                log_rate_ratio(vector, inputs, layout, static)[0], static
            )

        correction = godambe_correction(
            final_objective,
            theta,
            indices=indices,
            deterministic_objective=deterministic,
            block_score=_year_block_score(objective_with, theta, indices, static_builder),
            n_bootstrap=n_bootstrap,
            n_blocks=len(inputs.years),
            seed=seed,
        )
        covariance = expand_covariance(correction, layout.size)
        draws = draw_parameters(theta, covariance, n_draws=n_draws, seed=seed + 1)

    stage_seconds["correction"] = time.time() - stage_start
    stage_start = time.time()

    anchors = {"anchor": [anchor_station]}
    diagnostics = graph_diagnostics.observer_diagnostics(inputs.graph, anchors)
    diagnostics = diagnostics.rename(columns={"anchored_anchor": "anchored_to_anchor"})

    series = combine_series(theta, inputs, layout, static, draws=draws)
    stage_seconds["series"] = time.time() - stage_start
    stage_start = time.time()
    observers = observer_table(
        theta,
        inputs,
        layout,
        static,
        covariance=covariance,
        cov_indices=correction.indices if correction is not None else None,
        diagnostics=diagnostics[
            ["station", "component", "degree", "on_bridge", "anchored", "fingerprint_only"]
        ],
    )

    stage_seconds["observers"] = time.time() - stage_start
    empirical_share = float(
        np.mean(
            (inputs.activity >= detection.meta.get("coverage_lo", -np.inf))
            & (inputs.activity <= detection.meta.get("coverage_hi", np.inf))
        )
    )
    meta = {
        "name": name,
        "kernel": kernel,
        "n_outer": len(passes),
        "fingerprint_weight": fingerprint_weight,
        "lambda_penalty": lambda_penalty,
        "drift_log_sigma_mean": drift_log_sigma_mean,
        "n_parameters": layout.size,
        "n_bootstrap": n_bootstrap if correct else 0,
        "activity_years_measured": empirical_share,
        "detection_source": detection.meta.get("source"),
        "stage_seconds": stage_seconds,
        "total_seconds": float(sum(stage_seconds.values())),
        "activity_map": activity_map.as_row(),
        **inputs.meta,
    }
    if correction is not None:
        meta["median_inflation"] = correction.median_inflation
        meta["n_weak_modes"] = correction.n_weak_modes
        meta.update({f"correction_{k}": v for k, v in correction.meta.items()})

    return NetworkFit(
        theta=theta,
        inputs=inputs,
        layout=layout,
        static=static,
        series=series,
        observers=observers,
        correction=correction,
        draws=draws,
        passes=passes,
        meta=meta,
    )


def _refresh_activity(
    inputs: NetworkInputs,
    series: pd.DataFrame,
    activity_map: ActivityMap,
    detection: DetectionGrid,
) -> NetworkInputs:
    """One pass of the fixed point: the new series implies a new activity level."""
    smoothed = smooth_counts(series["date_ord"].to_numpy(), series["gn"].to_numpy())
    by_year = (
        pd.DataFrame(
            {
                "year": np.floor(decimal_year(smoothed.index.to_numpy())).astype(np.int64),
                "gn": smoothed.to_numpy(),
            }
        )
        .groupby("year")["gn"]
        .mean()
        .reindex(inputs.years)
    )
    updated = np.asarray(activity_map(by_year.to_numpy()), dtype=float)
    fallback = float(activity_map(np.array(1.0)))
    updated = np.where(np.isfinite(updated) & (updated > 0), updated, fallback)
    inputs.activity = np.clip(updated, detection.activity.min(), detection.activity.max())
    return inputs


# ---------------------------------------------------------------------------
# A matched simulator, for recovery tests only
# ---------------------------------------------------------------------------


def simulate_matched_network(
    detection: DetectionGrid,
    *,
    n_observers: int = 40,
    n_days: int = 4000,
    first_ord: int = 700_000,
    cycle_days: float = 4015.0,
    gn_min: float = 0.4,
    gn_max: float = 6.0,
    threshold_lo: float = 8.0,
    threshold_hi: float = 160.0,
    observing_fraction: float = 0.45,
    career_fraction: float = 0.5,
    drift_sigma: float = 0.0,
    split_rate: float = 0.0,
    anchor_station: int = ANCHOR_STATION,
    activity_map: ActivityMap | None = None,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A network generated by *this* model, for recovery tests.

    **This is not a benchmark and must never be reported as one.**  It is matched to
    the estimator by construction, so it tests the solver and the parameterization
    and nothing else.  The research corpus is blunt about the distinction:
    "benchmarks that reuse the same generative model as the inference method will
    overstate performance", which is why Step 2's blind benchmark exists and is
    deliberately misspecified in four stated ways.

    What a matched simulation *is* good for is the SDSS test: their noise-free
    simulations reached about 0.1%, "so the residual error came from model
    misspecification rather than from statistics", and the report's conclusion is
    that "calibration quality is therefore limited by model misspecification, not by
    solver or statistics".  Recovery here bounds the solver's own contribution, so
    whatever the blind benchmark leaves over is attributable elsewhere.

    Returns ``(observations, truth)`` where ``truth`` carries each station's true
    threshold and the latent daily count.
    """
    from gn_next_gen.models.emissions import empirical_count_pmf

    rng = np.random.default_rng(seed)
    days = first_ord + np.arange(n_days)
    phase = 2.0 * np.pi * np.arange(n_days) / cycle_days
    level = gn_min + 0.5 * (gn_max - gn_min) * (1.0 - np.cos(phase))

    # Activity through the activity map, and the latent count from the reference's
    # own activity-conditional distribution -- the same two objects the estimator
    # reads.  Drawing the latent count from a plain Poisson instead makes this test
    # quietly *unmatched*, and the occupancy term then shows a threshold bias of
    # +0.76 in log that belongs to the mismatch rather than to the solver.  Being
    # matched is the whole point here; anything unmatched belongs in the blind
    # benchmark.
    if activity_map is None:
        activity = np.clip(
            np.interp(
                level,
                [gn_min, gn_max],
                [detection.activity_ref * 0.3, detection.activity_ref * 2.0],
            ),
            detection.activity.min(),
            detection.activity.max(),
        )
    else:
        activity = np.clip(
            np.asarray(activity_map(level), dtype=float),
            detection.activity.min(),
            detection.activity.max(),
        )

    pmf, pmf_activity = empirical_count_pmf(
        detection.survey,
        reference_threshold_msh=detection.reference_threshold_msh,
    )
    bin_of = np.argmin(np.abs(activity[:, None] - pmf_activity[None, :]), axis=1)
    counts_axis = np.arange(pmf.shape[1])
    latent = np.array([rng.choice(counts_axis, p=pmf[b]) for b in bin_of], dtype=np.int64)

    thresholds = np.exp(rng.uniform(np.log(threshold_lo), np.log(threshold_hi), n_observers))
    rows = []
    truth_rows = []
    for index in range(n_observers):
        station = 100 + index
        span = max(int(career_fraction * n_days), 50)
        start = rng.integers(0, max(n_days - span, 1))
        window = np.arange(start, start + span)
        observed = window[rng.random(len(window)) < observing_fraction]
        if len(observed) < 80:
            observed = window[:80]

        walk = np.zeros(len(observed))
        if drift_sigma > 0:
            years = (observed - observed[0]) // 365
            steps = rng.normal(0.0, drift_sigma, int(years.max()) + 1)
            walk = np.cumsum(steps)[years]
        threshold_t = thresholds[index] * np.exp(walk)
        log_k = detection(threshold_t, activity[observed])
        k = np.exp(np.asarray(log_k))

        seen = rng.binomial(latent[observed], np.minimum(k, 1.0))
        extra = rng.poisson(np.maximum(latent[observed] * (k - 1.0), 0.0))
        reported = seen + extra
        if split_rate > 0:
            reported = reported + rng.binomial(reported, split_rate)

        rows.append(
            pd.DataFrame(
                {
                    "date_ord": days[observed],
                    "station": station,
                    "groups": reported,
                }
            )
        )
        truth_rows.append(
            {
                "station": station,
                "true_threshold_msh": float(thresholds[index]),
                "n_days": int(len(observed)),
                "mean_k": float(np.mean(k)),
            }
        )

    anchor_days = np.arange(n_days)[rng.random(n_days) < 0.9]
    rows.append(
        pd.DataFrame(
            {
                "date_ord": days[anchor_days],
                "station": anchor_station,
                "groups": latent[anchor_days],
            }
        )
    )
    truth_rows.append(
        {
            "station": anchor_station,
            "true_threshold_msh": float(detection.reference_threshold_msh),
            "n_days": int(len(anchor_days)),
            "mean_k": 1.0,
        }
    )

    observations = pd.concat(rows, ignore_index=True)
    truth = pd.DataFrame(truth_rows)
    truth.attrs["latent"] = pd.Series(latent, index=days)
    truth.attrs["activity"] = pd.Series(activity, index=days)
    return observations, truth
