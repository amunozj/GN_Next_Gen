"""Invariants of the KMAS + SWPC observer-count extension.

The extension is what carries the observation record past v1.3's 2010-12-31 end,
so the checks here are mostly about it being an honest reduction of the survey
catalogues: counts that fall as the threshold rises, zero-but-observed kept
distinct from not-observed, and no area column smuggled in behind the decision to
keep SWPC areas out of the reference (D13).
"""

import datetime as dt
import json

import pandas as pd
import pytest

from gn_next_gen.paths import (
    GN_EXTENSION_COUNTS,
    GN_EXTENSION_STATIONS,
    GN_OBSERVER_EXTENSION,
    REFERENCE_DAY_STATUS,
)

V13_LAST_DAY = dt.date(2010, 12, 31).toordinal()
STATIONS = {"kmas": 1006, "swpc": 1008}
SPANS = {"kmas": (1954, 2026), "swpc": (1996, 2026)}


@pytest.fixture(scope="module")
def counts():
    return pd.read_parquet(GN_EXTENSION_COUNTS)


@pytest.fixture(scope="module")
def stations():
    return pd.read_csv(GN_EXTENSION_STATIONS)


@pytest.fixture(scope="module")
def provenance():
    return json.loads((GN_OBSERVER_EXTENSION / "provenance.json").read_text())


def test_only_kmas_and_swpc(counts, stations):
    assert set(counts["survey"]) == set(STATIONS)
    assert dict(zip(stations["survey"], stations["station"])) == STATIONS


def test_station_codes_cannot_collide_with_v13(counts):
    """v1.3 uses 1-775; these are upstream's own survey tags."""
    assert (counts["station"] > 775).all()


def test_no_area_column_leaks_in(counts, provenance):
    """SWPC areas are deliberately absent from data/raw (D13)."""
    assert not [c for c in counts.columns if "area" in c.lower()]
    assert provenance["contains_areas"] is False


def test_spans_and_extension_flag(counts):
    for survey, (first, last) in SPANS.items():
        part = counts[counts["survey"] == survey]
        assert part["year"].min() == first
        assert part["year"].max() >= last
        # Both halves must be non-empty: the pre-2010 overlap is what calibrates
        # the new observer against the existing network.
        assert (~part["extends_v13"]).any(), f"{survey} has no pre-2010 overlap"
        assert part["extends_v13"].any(), f"{survey} adds nothing past 2010"

    assert (counts.loc[counts["extends_v13"], "date_ord"] > V13_LAST_DAY).all()
    assert (counts.loc[~counts["extends_v13"], "date_ord"] <= V13_LAST_DAY).all()


def test_one_row_per_survey_day_threshold(counts):
    assert not counts.duplicated(["survey", "date_ord", "threshold_msh"]).any()


def test_thresholds_include_the_sweep_grid_and_each_anchor(counts, stations):
    anchors = dict(zip(stations["survey"], stations["anchor_threshold_msh"]))
    for survey, part in counts.groupby("survey"):
        thresholds = set(part["threshold_msh"].unique())
        assert {0.0, 10.0, 30.0, 60.0, 100.0} <= thresholds
        assert anchors[survey] in thresholds


def test_counts_are_monotone_non_increasing_in_threshold(counts):
    """The strongest single check that the thresholding is right."""
    for col in ("groups_native", "groups_dbcal"):
        wide = counts.pivot_table(
            index=["survey", "date_ord"], columns="threshold_msh", values=col
        )
        ordered = wide[sorted(wide.columns)].to_numpy(dtype="float64")
        diffs = ordered[:, 1:] - ordered[:, :-1]
        import numpy as np

        assert np.all(np.isnan(diffs) | (diffs <= 0)), f"{col} rises with the threshold"


def test_day_status_semantics(counts):
    assert set(counts["day_status"].unique()) <= {0, 1, 2}
    no_obs = counts[counts["day_status"] == 0]
    assert no_obs["groups_native"].isna().all(), "no observation must not look like zero"
    assert (no_obs["groups_v13_convention"] == -1).all(), "v1.3's own missing sentinel"

    spotless = counts[counts["day_status"] == 1]
    assert (spotless["groups_native"] == 0).all()
    assert (spotless["groups_v13_convention"] == 0).all()

    active = counts[(counts["day_status"] == 2) & (counts["threshold_msh"] == 0.0)]
    assert (active["groups_native"] > 0).all()


def test_zero_threshold_counts_match_the_reference_day_status(counts):
    """Ties the extension to workstream A: at T=0, KMAS counts are its n_groups."""
    kmas = counts[(counts["survey"] == "kmas") & (counts["threshold_msh"] == 0.0)]
    ref = pd.read_parquet(REFERENCE_DAY_STATUS)
    ref = ref[ref["survey"] == "kmas"][["date_ord", "n_groups", "day_status"]]
    merged = kmas.merge(ref, on="date_ord", suffixes=("", "_ref"))
    assert len(merged) == len(ref), "the two tables must cover the same KMAS days"
    assert (merged["day_status"] == merged["day_status_ref"]).all()
    observed = merged[merged["day_status"] > 0]
    assert (observed["groups_native"] == observed["n_groups_ref"]).all()


def test_roles_record_the_single_use_rule(counts, stations, provenance):
    roles = dict(zip(stations["survey"], stations["role"]))
    assert roles["kmas"] == "reference_or_observer_choose_one"
    assert roles["swpc"] == "observer_counts_only"
    assert set(counts.loc[counts["survey"] == "kmas", "role"]) == {roles["kmas"]}

    rule = provenance["single_use_rule"].lower()
    assert "never both" in rule
    assert "kmas" in rule
    readme = (GN_OBSERVER_EXTENSION / "README.md").read_text().lower()
    assert "never both" in readme


def test_firewall_lineage_statement(provenance):
    lineage = provenance["data_lineage"].lower()
    for forbidden in ("isotope", "aurora", "proxy", "silso"):
        assert forbidden in lineage
    assert "no isotope" in lineage
