"""Rotation kernel: why the reference's group "lifetimes" stop at about 13 days.

``obs_duration`` in the reference snapshot is the observed **span** of a group, and
it is censored by disk passage: RGO has 1,976 groups at 13 days, 343 at 14, 25 at 15
and a maximum of 22.  Its empirical survival ratio sits at 0.87-0.89 for days 1-9
and then collapses (0.567 at day 11, 0.158 at day 12).  A geometric chain cannot
produce that cliff, so a comparison of simulated against observed lifetimes is only
like-for-like if the simulation rotates groups off the limb the way the Sun does.

The generative model here is the marked point process of the research report's
"M/G/infinity inference" section: emergences in time and Carrington longitude, each
group carrying an intrinsic lifetime, seen only while within 90 degrees of the
central meridian.  Occupancy is then Poisson shot noise with a 0/1 kernel, and
Campbell's theorem predicts the signature Check A looks for -- a fast autocovariance
decay over ~13.5 days, a trough, and a recurrence bump near 27 days for long-lived
groups.  That bump is present in RGO at T = 30 MSH (lag-27 ACF 0.736 against 0.640
at lag 13), which is why this module exists rather than a geometric lifetime fit.

Simulation uses NumPy generators rather than JAX keys: nothing here is
differentiated, and the caller seeds it from the run config.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Synodic solar rotation period in days.  A group is visible for at most half of
#: one rotation, which is the ~13.6-day ceiling the reference shows.
SYNODIC_ROTATION_DAYS = 27.2753

#: Longitude half-width of the visible disk, in degrees.
VISIBLE_HALF_WIDTH_DEG = 90.0

#: Lifetimes are simulated out to this many days; beyond it the visible span is
#: already saturated by disk passage, so the cap cannot bias the comparison.
MAX_LIFETIME_DAYS = 150


@dataclass
class MarkedSimulation:
    """One realization of the marked process.

    Attributes
    ----------
    daily_counts
        Groups visible on each day of the grid.
    visible_spans
        Per group that was visible at least once: last visible day - first + 1,
        counting every apparition.
    first_apparition_spans
        The same, but for the **first contiguous** visible window only. This is the
        quantity to compare against the snapshot's ``obs_duration``: the surveys give
        a group that rotates out and returns a *new* identifier, which is why RGO's
        maximum span is 22 days while an unrestricted simulation reaches ~90.
    visible_days
        Per such group: how many days it was actually seen, which differs from the
        span whenever a group is seen, rotates out, and returns.
    intrinsic_lifetimes
        The latent lifetimes, including those of groups never seen at all.
    n_never_visible
        Groups that lived entirely on the far side. Not a loss of realism but the
        point: the reference cannot see them either.
    """

    daily_counts: np.ndarray
    visible_spans: np.ndarray
    first_apparition_spans: np.ndarray
    visible_days: np.ndarray
    intrinsic_lifetimes: np.ndarray
    n_never_visible: int
    n_recurrent: int = 0


def visible_fraction_of_rotation() -> float:
    """Fraction of a rotation a group spends on the visible hemisphere."""
    return 2.0 * VISIBLE_HALF_WIDTH_DEG / 360.0


def max_transit_days() -> float:
    """Longest unbroken visibility for a sufficiently long-lived group."""
    return SYNODIC_ROTATION_DAYS * visible_fraction_of_rotation()


def expected_visible_span(intrinsic_mean_days: float, n: int = 200_000, seed: int = 0) -> float:
    """Monte-Carlo mean visible span for an exponential intrinsic lifetime.

    Used to invert the relationship: Check A fits ``phi`` to the *counts*, and this
    says what intrinsic lifetime that implies once rotation is accounted for.
    """
    rng = np.random.default_rng(seed)
    lifetimes = rng.exponential(intrinsic_mean_days, n)
    longitudes = rng.uniform(0.0, 360.0, n)
    _, _, _, first_spans, _ = _visible_windows(lifetimes, longitudes)
    seen = first_spans > 0
    return float(first_spans[seen].mean()) if seen.any() else 0.0


def _wrap_degrees(angle: np.ndarray) -> np.ndarray:
    """Map degrees to (-180, 180]."""
    return (angle + 180.0) % 360.0 - 180.0


def _visible_windows(
    lifetimes: np.ndarray, longitudes: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per group: span, visible-day count, the day mask, the first-apparition span.

    A group emerges at Carrington longitude ``longitudes`` measured from the central
    meridian and rotates westward at ``360 / SYNODIC_ROTATION_DAYS`` degrees a day.
    It is visible on day ``d`` when its wrapped longitude is within the half-width.

    Returns ``(spans, visible_day_counts, mask, first_apparition_spans, recurrent)``
    where ``recurrent`` flags the groups seen in more than one apparition -- the ones
    a survey would have split into separate identifiers.
    """
    capped = np.minimum(lifetimes, MAX_LIFETIME_DAYS)
    horizon = int(np.ceil(capped.max())) + 1 if len(capped) else 1
    offsets = np.arange(horizon)[None, :]

    alive = offsets <= capped[:, None]
    theta = _wrap_degrees(longitudes[:, None] + 360.0 * offsets / SYNODIC_ROTATION_DAYS)
    visible = alive & (np.abs(theta) <= VISIBLE_HALF_WIDTH_DEG)

    any_visible = visible.any(axis=1)
    first = np.where(any_visible, visible.argmax(axis=1), 0)
    last = np.where(any_visible, horizon - 1 - visible[:, ::-1].argmax(axis=1), -1)
    spans = np.where(any_visible, last - first + 1, 0)

    # The first apparition ends at the first day the group is not visible after it
    # has been seen; a later return is a separate apparition and, in a survey, a
    # separate group.
    seen_then_gone = any_visible & (visible.sum(axis=1) < spans)
    padded = np.concatenate([visible, np.zeros((visible.shape[0], 1), dtype=bool)], axis=1)
    rows = np.arange(visible.shape[0])
    after_first = padded.copy()
    offsets = np.arange(padded.shape[1])[None, :]
    after_first &= offsets >= first[:, None]
    gap = (~after_first) & (offsets >= first[:, None])
    first_gap = np.where(gap.any(axis=1), gap.argmax(axis=1), padded.shape[1])
    first_spans = np.where(any_visible, first_gap - first, 0)
    del rows
    return spans, visible.sum(axis=1), visible, first_spans, seen_then_gone


def simulate_marked_process(
    lam_t: np.ndarray,
    intrinsic_mean_days: float,
    *,
    seed: int = 0,
    lifetime: str = "exponential",
) -> MarkedSimulation:
    """Simulate emergences, rotation and visibility on a daily grid.

    Parameters
    ----------
    lam_t
        Daily emergence rate, on the same grid as the observations.  This is the
        *emergence* rate of new groups anywhere on the Sun, so the visible count it
        produces is lower than ``lam_t / (1 - phi)`` by the visible fraction.
    intrinsic_mean_days
        Mean intrinsic lifetime, before rotation censors it.
    lifetime
        ``"exponential"`` (memoryless, the chain's own assumption) or ``"weibull"``
        with shape 1.5, a heavier-shouldered alternative for sensitivity.
    """
    rng = np.random.default_rng(seed)
    lam_t = np.asarray(lam_t, dtype=float)
    n_days = len(lam_t)

    births = rng.poisson(lam_t)
    emergence_day = np.repeat(np.arange(n_days), births)
    n_groups = len(emergence_day)
    if n_groups == 0:
        empty = np.array([])
        return MarkedSimulation(np.zeros(n_days, dtype=int), empty, empty, empty, empty, 0, 0)

    if lifetime == "exponential":
        lifetimes = rng.exponential(intrinsic_mean_days, n_groups)
    elif lifetime == "weibull":
        from math import gamma

        shape = 1.5
        scale = intrinsic_mean_days / gamma(1 + 1 / shape)  # so the mean is as asked
        lifetimes = scale * rng.weibull(shape, n_groups)
    else:
        raise ValueError(f"unknown lifetime model {lifetime!r}")

    longitudes = rng.uniform(0.0, 360.0, n_groups)
    spans, n_visible, visible, first_spans, recurrent = _visible_windows(lifetimes, longitudes)

    rows, cols = np.nonzero(visible)
    days = emergence_day[rows] + cols
    inside = days < n_days
    daily_counts = np.bincount(days[inside], minlength=n_days)

    seen = spans > 0
    return MarkedSimulation(
        daily_counts=daily_counts,
        visible_spans=spans[seen],
        first_apparition_spans=first_spans[seen],
        visible_days=n_visible[seen],
        intrinsic_lifetimes=lifetimes,
        n_never_visible=int((~seen).sum()),
        n_recurrent=int(recurrent.sum()),
    )
