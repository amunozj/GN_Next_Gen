"""Daily group counts per observer, 1610-2026, and the observer table.

Two sources, concatenated on request:

* ``gn_database_v1.3/`` -- the revised collection, 1610-2010, release **v1.3_4**
  (ADR-010).  ``STATION = 0`` marks a day with no observation and ``GROUPS = -1``
  marks the missing count that goes with it; neither survives loading.
* ``gn_observer_extension_2026-10/`` -- KMAS and SWPC daily counts to 2026, which
  carry the record past v1.3's end.  **KMAS is dual-use**: a run declares it as
  reference areas *or* observer counts, never both (ADR-009), and asking for its
  counts while it is the reference raises here rather than quietly double-counting.

**D16** -- a few ``(date, station)`` keys carry more than one row, so the loader
resolves them explicitly and records which rule it used.  In release v1.3_4 this is
3 keys and one real conflict (Takuma, 1982-10-13, 8 against 7); in v1.3_3 it was 217
keys and 95 conflicts.  The default rule is ``max``: 8 matches that observer's
neighbouring days (9, 7, 8), and the alternatives stay available as sensitivities.
"""

from __future__ import annotations

import pandas as pd

from gn_next_gen.io.cache import cached_frame
from gn_next_gen.paths import (
    GN_EXTENSION_COUNTS,
    GN_V13_CURRENT_RELEASE,
    GN_V13_OBSERVATIONS,
    GN_V13_OBSERVERS,
)

DUPLICATE_RULES = ("max", "drop_conflicting", "first", "keep_both")

#: Station codes of the extension surveys (upstream's own tags; cannot collide
#: with v1.3's 1-776).
EXTENSION_STATIONS = {"kmas": 1006, "swpc": 1008}

#: Each extension survey's own anchor threshold from the upstream joint posterior.
ANCHOR_THRESHOLDS = {"kmas": 23.41, "swpc": 20.68}

OBSERVATION_COLUMNS = ["date_ord", "station", "groups", "source"]


def resolve_duplicate_keys(
    frame: pd.DataFrame,
    rule: str = "max",
    *,
    key: tuple[str, str] = ("date_ord", "station"),
    value: str = "groups",
) -> pd.DataFrame:
    """Collapse duplicated observer-days under one of the D16 rules.

    Parameters
    ----------
    rule
        ``"max"`` keeps the largest count on a conflicting key (the default: a
        recount that found more groups is the likelier reading, and it matches the
        one surviving conflict's neighbouring days).  ``"drop_conflicting"`` keeps
        identical duplicates once and drops conflicting observer-days entirely, on
        the grounds that we cannot tell which row is the recount.  ``"first"`` keeps
        the earlier row in file order -- which is date-sorted in v1.3_4, so "first"
        is not "original".  ``"keep_both"`` leaves them, which puts the same
        observer-day into the likelihood twice and exists only for sensitivity runs.

    Notes
    -----
    Identical duplicates are collapsed by every rule except ``keep_both``: they
    carry no information beyond the single value.
    """
    if rule not in DUPLICATE_RULES:
        raise ValueError(f"unknown duplicate rule {rule!r}; expected one of {DUPLICATE_RULES}")
    if rule == "keep_both":
        return frame.reset_index(drop=True)

    keys = list(key)
    dup_mask = frame.duplicated(keys, keep=False)
    if not dup_mask.any():
        return frame.reset_index(drop=True)

    singles = frame[~dup_mask]
    dups = frame[dup_mask]
    spread = dups.groupby(keys)[value].transform("nunique")
    identical = dups[spread == 1].drop_duplicates(keys, keep="first")
    conflicting = dups[spread > 1]

    if rule == "drop_conflicting":
        resolved = identical
    elif rule == "max":
        idx = conflicting.groupby(keys)[value].idxmax()
        resolved = pd.concat([identical, conflicting.loc[idx]])
    else:  # "first"
        resolved = pd.concat([identical, conflicting.drop_duplicates(keys, keep="first")])

    out = pd.concat([singles, resolved]).sort_values(keys, kind="stable")
    return out.reset_index(drop=True)


def _load_v13(release: str) -> pd.DataFrame:
    """The v1.3 observations with sentinel rows removed and columns renamed."""
    if release != GN_V13_CURRENT_RELEASE:
        raise ValueError(
            f"release {release!r} is not the current release "
            f"({GN_V13_CURRENT_RELEASE}); regenerate it with "
            f"`python scripts/convert_gn_database_v13.py --release {release}` and read "
            "it from data/interim/ explicitly (ADR-010)"
        )
    raw = pd.read_parquet(GN_V13_OBSERVATIONS, columns=["ORDINAL", "STATION", "GROUPS"])
    real = raw[raw["STATION"] > 0]
    if not (real["GROUPS"] >= 0).all():
        raise AssertionError("GROUPS = -1 must only ever accompany STATION = 0")
    out = real.rename(columns={"ORDINAL": "date_ord", "STATION": "station", "GROUPS": "groups"})
    out["source"] = release
    return out[OBSERVATION_COLUMNS]


def _load_extension(
    kmas_role: str,
    threshold: str | float,
    post_v13_only: bool,
) -> pd.DataFrame:
    """KMAS/SWPC daily counts as observers, honouring the single-use rule."""
    counts = pd.read_parquet(
        GN_EXTENSION_COUNTS,
        columns=[
            "survey",
            "station",
            "date_ord",
            "threshold_msh",
            "groups_v13_convention",
            "extends_v13",
        ],
    )
    if kmas_role == "reference":
        counts = counts[counts["survey"] != "kmas"]
    elif kmas_role != "observer_counts":
        raise ValueError(f"kmas_role must be 'reference' or 'observer_counts', got {kmas_role!r}")

    frames = []
    for survey, group in counts.groupby("survey", sort=True):
        want = ANCHOR_THRESHOLDS[str(survey)] if threshold == "anchor" else float(threshold)
        chosen = group[group["threshold_msh"].round(2) == round(want, 2)]
        if chosen.empty:
            available = sorted(group["threshold_msh"].unique())
            raise ValueError(f"threshold {want} not in {survey} extension; have {available}")
        frames.append(chosen)

    if not frames:
        return pd.DataFrame(columns=OBSERVATION_COLUMNS)

    ext = pd.concat(frames, ignore_index=True)
    if post_v13_only:
        ext = ext[ext["extends_v13"]]
    ext = ext[ext["groups_v13_convention"] >= 0]  # -1 = not observed
    ext = ext.rename(columns={"groups_v13_convention": "groups"})
    ext["source"] = "extension_" + ext["survey"].astype(str)
    return ext[OBSERVATION_COLUMNS]


def load_observations(
    release: str = GN_V13_CURRENT_RELEASE,
    *,
    duplicate_rule: str = "max",
    include_extension: bool = False,
    kmas_role: str = "reference",
    extension_threshold: str | float = "anchor",
    extension_post_v13_only: bool = True,
    use_cache: bool = True,
    force: bool = False,
) -> pd.DataFrame:
    """Long-format daily group counts: ``date_ord, station, groups, source``.

    Parameters
    ----------
    release
        v1.3 release to read.  Only the current one lives in ``data/raw`` (ADR-010).
    duplicate_rule
        How to resolve duplicated ``(date, station)`` keys; see
        :func:`resolve_duplicate_keys` and issue D16.
    include_extension
        Append KMAS/SWPC daily counts so the record reaches 2026.
    kmas_role
        ``"reference"`` (default) keeps KMAS for the reference survey and therefore
        **excludes** its observer counts; ``"observer_counts"`` does the opposite.
        The single-use rule of ADR-009, enforced here as well as in the run config.
    extension_threshold
        ``"anchor"`` uses each survey's own anchor threshold (KMAS 23.41,
        SWPC 20.68 MSH), which is the scale the renovation calibrated against the
        v1.3 network; a number selects one of the sweep thresholds instead.
    extension_post_v13_only
        Keep only the rows after 2010-12-31.  The pre-2010 overlap is a calibration
        diagnostic, not a second copy of the network's own observations.
    """
    options = {
        "release": release,
        "duplicate_rule": duplicate_rule,
        "include_extension": include_extension,
        "kmas_role": kmas_role,
        "extension_threshold": extension_threshold,
        "extension_post_v13_only": extension_post_v13_only,
    }
    inputs = [GN_V13_OBSERVATIONS]
    if include_extension:
        inputs.append(GN_EXTENSION_COUNTS)

    def build() -> pd.DataFrame:
        frame = _load_v13(release)
        if include_extension:
            frame = pd.concat(
                [frame, _load_extension(kmas_role, extension_threshold, extension_post_v13_only)],
                ignore_index=True,
            )
        frame = resolve_duplicate_keys(frame, duplicate_rule)
        return frame.astype({"date_ord": "int64", "station": "int64", "groups": "int64"})

    return cached_frame("observations", inputs, options, build, use_cache=use_cache, force=force)


def load_observers(release: str = GN_V13_CURRENT_RELEASE) -> pd.DataFrame:
    """The listed observers: ``STATION, INITIAL, FINAL, TOT_OBS, OBSERVER, changed_in_v13``.

    ``INITIAL``/``FINAL`` are nullable integers: null for the placeholder observers
    that carry no observations (34 in v1.3_4), which v1.3_3 wrote as the string
    ``'NA'`` (D18).
    """
    if release != GN_V13_CURRENT_RELEASE:
        raise ValueError(f"release {release!r} is not the current release")
    return pd.read_parquet(GN_V13_OBSERVERS)
