"""The reference survey: per-group, per-day catalogues, 1825-2026 (ADR-009).

``area_corr`` is the model-facing area -- native, whole-spot,
foreshortening-corrected, MSH.  ``area_corr_dbcal*`` is the same on the common
Debrecen scale and is a **prior and cross-check only**, never a hard
pre-correction, or the upstream calibration is applied twice (ADR-003, D5).

Two exclusions are defaults here rather than caller discipline:

* **Spörer and Schwabe are out of Step 1** (PI, 2026-11-21).  Spörer carries
  ``obs_duration = 1`` for every one of its 5,421 groups -- it tracks no group
  across days, so it yields no lifetimes -- and neither survey records a single
  spotless day (``day_status == 1`` is empty for both, issue **D23**), so neither
  can contribute the zero-fraction or run-length statistics Check A rests on.
  ADR-009 still defines the snapshot as five surveys; this is about use, not
  contents, and the full set is one argument away.
* **D22** -- KMAS reuses a ``group_id`` for physically distinct groups on 414
  group-days (289 ids).  Daily counts are unaffected because they count *rows*,
  which is the correct unit, but anything keyed on ``group_id`` is wrong: those
  groups' apparent lifetimes merge two apparitions (median 11 days against 5).
  :func:`load_group_single_entry` therefore drops them by default.
"""

from __future__ import annotations

import pandas as pd

from gn_next_gen.paths import (
    REFERENCE_DAY_STATUS,
    REFERENCE_GROUP_DAY,
    REFERENCE_GROUP_SINGLE,
)

#: Every survey in the snapshot (ADR-009).
CORE_FIVE = ("schwabe", "sporer", "rgo", "debrecen", "kmas")

#: The surveys Step 1 uses: RGO and Debrecen are the two anchors, KMAS bridges
#: them through cycle 20.  See the module docstring for why the other two are out.
STEP1_SURVEYS = ("rgo", "debrecen", "kmas")

#: ``day_status`` values, which replace the old ``AREA1`` NaN/0 convention (D10).
DAY_NO_OBSERVATION, DAY_SPOTLESS, DAY_ACTIVE = 0, 1, 2

GROUP_DAY_COLUMNS = ["survey", "group_id", "date_ord", "area_corr"]


def _check_surveys(surveys: tuple[str, ...] | list[str]) -> list[str]:
    unknown = set(surveys) - set(CORE_FIVE)
    if unknown:
        raise ValueError(f"unknown survey(s) {sorted(unknown)}; expected {CORE_FIVE}")
    return list(surveys)


def kmas_colliding_group_ids() -> set[str]:
    """The 289 KMAS ``group_id``s that cover more than one physical group (D22)."""
    kmas = pd.read_parquet(REFERENCE_GROUP_DAY, columns=["survey", "group_id", "date_ord"])
    kmas = kmas[kmas["survey"] == "kmas"]
    per_key = kmas.groupby(["group_id", "date_ord"]).size()
    return set(per_key[per_key > 1].index.get_level_values("group_id").unique())


def load_group_days(
    surveys: tuple[str, ...] | list[str] = STEP1_SURVEYS,
    *,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """One row per (survey, group, day) -- the reference.

    Every row is kept, including the D22 collisions: **the row is the unit of
    observation**, so daily counts taken from this table are correct.
    """
    surveys = _check_surveys(surveys)
    frame = pd.read_parquet(REFERENCE_GROUP_DAY, columns=columns or GROUP_DAY_COLUMNS)
    return frame[frame["survey"].isin(surveys)].reset_index(drop=True)


def load_day_status(
    surveys: tuple[str, ...] | list[str] = STEP1_SURVEYS,
) -> pd.DataFrame:
    """One row per (survey, day): ``day_status``, ``n_groups`` and the daily areas.

    ``area_total`` is the whole-Sun daily sunspot area and ``area_north``/``area_south``
    its hemispheric parts; :func:`gn_next_gen.reference.activity_level` smooths them
    into the activity level the group-area law depends on.  The hemispheric columns are
    **diagnostic only** -- hemisphere assignment is positional, which ADR-004 bars as a
    model input.
    """
    surveys = _check_surveys(surveys)
    frame = pd.read_parquet(
        REFERENCE_DAY_STATUS,
        columns=[
            "survey",
            "date_ord",
            "day_status",
            "n_groups",
            "area_total",
            "area_north",
            "area_south",
        ],
    )
    return frame[frame["survey"].isin(surveys)].reset_index(drop=True)


def load_group_single_entry(
    surveys: tuple[str, ...] | list[str] = STEP1_SURVEYS,
    *,
    drop_kmas_collisions: bool = True,
) -> pd.DataFrame:
    """One row per (survey, group), carrying ``obs_duration``.

    ``obs_duration`` is the observed **span** (last day - first day + 1), not the
    number of days the group was seen: about 10% of RGO groups have an interior gap.
    It is also censored by disk passage, so it measures *visible residence* and not
    intrinsic lifetime -- RGO has 1,976 groups at 13 days, 343 at 14 and 25 at 15.
    """
    surveys = _check_surveys(surveys)
    frame = pd.read_parquet(
        REFERENCE_GROUP_SINGLE,
        columns=[
            "survey",
            "group_id",
            "date_ord",
            "obs_duration",
            "group_obs_kind",
            "area_corr",
            "area_corr_unc",
            "cycle",
            "hemisphere",
        ],
    )
    frame = frame[frame["survey"].isin(surveys)]
    if drop_kmas_collisions:
        bad = kmas_colliding_group_ids()
        frame = frame[~((frame["survey"] == "kmas") & (frame["group_id"].isin(bad)))]
    return frame.reset_index(drop=True)


def daily_counts(
    threshold_msh: float = 0.0,
    surveys: tuple[str, ...] | list[str] = STEP1_SURVEYS,
    *,
    group_days: pd.DataFrame | None = None,
    day_status: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Groups above an acuity threshold per (survey, observed day).

    The count is the number of **rows** for that (survey, day) with ``area_corr``
    *strictly greater* than ``threshold_msh`` -- the legacy convention, ported from
    ``legacy/ADFSSN/scripts/SSN_ADF.py`` l. 837-841 as an idea, not as a shape: the
    reference is long now, so this is a row count and "no observation" comes from
    ``day_status``, not from a NaN in ``AREA1`` (see ``docs/03_legacy/reuse_map.md``).

    Returns
    -------
    DataFrame
        ``survey, date_ord, day_status, n_groups`` for observed days only
        (``day_status > 0``), so a spotless day is a real zero and an unobserved day
        is absent rather than zero.
    """
    groups = load_group_days(surveys) if group_days is None else group_days
    days = load_day_status(surveys) if day_status is None else day_status

    observed = days[days["day_status"] > DAY_NO_OBSERVATION][["survey", "date_ord", "day_status"]]
    above = groups[groups["area_corr"] > threshold_msh]
    counted = above.groupby(["survey", "date_ord"]).size().rename("n_groups").reset_index()
    out = observed.merge(counted, on=["survey", "date_ord"], how="left")
    out["n_groups"] = out["n_groups"].fillna(0).astype("int64")
    out["threshold_msh"] = float(threshold_msh)
    return out.sort_values(["survey", "date_ord"], kind="stable").reset_index(drop=True)
