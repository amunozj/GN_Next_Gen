"""The run config must refuse the mistakes documentation has not prevented.

The firewall (ADR-002) and the KMAS single-use rule (ADR-009) were documentation-only
until this step. These tests are the point of the config layer: a tainted run has to
fail at load time, not produce a quietly overconfident posterior.

The circular-calibration rule joined them on 2026-11-23, when two more published GN
series were frozen under ``data/raw/`` -- one of them a *daily* reconstruction over
the same observers. Until then the folder's README claimed the validator refused
them and it did not: they would have been declared ``kind: sunspot`` and passed.
"""

from __future__ import annotations

import json

import pytest
import yaml

from gn_next_gen.config import (
    PROXY_INFORMED,
    SUNSPOT_ONLY,
    ConfigError,
    DatasetRef,
    RunConfig,
    lineage_record,
    load_run_config,
    validate_run_config,
    write_lineage,
)
from gn_next_gen.paths import CONFIGS

REFERENCE = DatasetRef("reference_groups_2026-10", "calibration", "sunspot")
EXTENSION = DatasetRef("gn_observer_extension_2026-10", "calibration", "sunspot")
SILSO_CAL = DatasetRef("silso", "calibration", "silso")
SILSO_CMP = DatasetRef("silso", "comparison", "silso")
PUBLISHED = "gn_published_series_2026-11"
SERIES_CAL = DatasetRef(PUBLISHED, "calibration", "published_series")
SERIES_CMP = DatasetRef(PUBLISHED, "comparison", "published_series")

STEP1_CONFIGS = ("step1_check_a.yaml", "step1_check_b.yaml")
STEP2_CONFIGS = ("step2_benchmark.yaml", "step2_baselines.yaml")


def test_committed_configs_load_and_validate():
    for name in STEP1_CONFIGS + STEP2_CONFIGS:
        config = load_run_config(CONFIGS / name)
        assert config.variant == SUNSPOT_ONLY
        assert config.seed, "every run config must carry a seed"
        assert config.kmas_role == "reference"
        assert "sporer" not in config.surveys and "schwabe" not in config.surveys


def test_committed_configs_keep_silso_out_of_calibration():
    for name in STEP1_CONFIGS:
        config = load_run_config(CONFIGS / name)
        for dataset in config.datasets_in_role("calibration"):
            assert dataset.kind == "sunspot", dataset


# --- the firewall (ADR-002) ------------------------------------------------


def test_firewall_rejects_silso_as_a_calibration_input():
    config = RunConfig(name="bad", variant=SUNSPOT_ONLY, datasets=(REFERENCE, SILSO_CAL))
    with pytest.raises(ConfigError, match="firewall"):
        validate_run_config(config)


@pytest.mark.parametrize("kind", ["isotope", "aurora", "proxy"])
def test_firewall_rejects_every_proxy_kind(kind):
    bad = DatasetRef("silso", "calibration", kind)  # path exists; the kind is the point
    with pytest.raises(ConfigError, match="firewall"):
        validate_run_config(RunConfig(name="bad", datasets=(REFERENCE, bad)))


def test_firewall_allows_silso_for_comparison():
    config = RunConfig(name="ok", datasets=(REFERENCE, SILSO_CMP))
    assert validate_run_config(config) is config


def test_proxy_informed_variant_may_calibrate_on_proxies():
    """The firewall is a property of the sunspot-only variant, not of the code."""
    config = RunConfig(name="proxy", variant=PROXY_INFORMED, datasets=(REFERENCE, SILSO_CAL))
    assert validate_run_config(config) is config


# --- no circular calibration (ADR-003) -------------------------------------


@pytest.mark.parametrize("variant", [SUNSPOT_ONLY, PROXY_INFORMED])
def test_no_variant_may_calibrate_on_a_published_series(variant):
    """Unlike the firewall, this one binds the proxy-informed variant too."""
    config = RunConfig(name="circular", variant=variant, datasets=(REFERENCE, SERIES_CAL))
    with pytest.raises(ConfigError, match="circular calibration"):
        validate_run_config(config)


def test_a_published_series_is_fine_for_comparison():
    config = RunConfig(name="ok", datasets=(REFERENCE, SERIES_CMP))
    assert validate_run_config(config) is config


def test_a_published_series_is_not_recorded_as_proxy_derived():
    """The two prohibitions stay apart in the lineage record.

    A published GN series is sunspot data. Folding it into FIREWALLED_KINDS would
    be the easy fix and would make the run's own firewall statement false.
    """
    record = lineage_record(RunConfig(name="circular", datasets=(REFERENCE, SERIES_CAL)))
    firewall = record["firewall"]
    assert firewall["proxy_derived_calibration_inputs"] == []
    assert "No isotope, aurora, proxy or SILSO input" in firewall["statement"]
    assert firewall["published_series_calibration_inputs"] == ["published_series"]
    assert "Calibration used published series" in firewall["circularity_statement"]


def test_a_clean_run_states_both_negatives():
    record = lineage_record(RunConfig(name="clean", datasets=(REFERENCE, SERIES_CMP)))
    firewall = record["firewall"]
    assert firewall["published_series_calibration_inputs"] == []
    assert firewall["circularity_statement"] == (
        "No published group-number series entered calibration."
    )


# --- the KMAS single-use rule (ADR-009) ------------------------------------


def test_kmas_may_not_be_reference_and_observer_counts_at_once():
    config = RunConfig(
        name="double",
        surveys=("rgo", "debrecen", "kmas"),
        kmas_role="observer_counts",
        datasets=(REFERENCE, EXTENSION),
    )
    with pytest.raises(ConfigError, match="single-use"):
        validate_run_config(config)


def test_kmas_as_reference_with_the_extension_present_is_fine():
    """SWPC carries no such constraint, so the extension itself is allowed."""
    config = RunConfig(
        name="ok",
        surveys=("rgo", "debrecen", "kmas"),
        kmas_role="reference",
        datasets=(REFERENCE, EXTENSION),
    )
    assert validate_run_config(config) is config


def test_kmas_as_observer_counts_requires_dropping_it_from_surveys():
    good = RunConfig(
        name="ok",
        surveys=("rgo", "debrecen"),
        kmas_role="observer_counts",
        datasets=(REFERENCE, EXTENSION),
    )
    assert validate_run_config(good) is good


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("variant", "whatever", "unknown variant"),
        ("kmas_role", "both", "kmas_role"),
        ("duplicate_rule", "mean", "duplicate_rule"),
    ],
)
def test_bad_scalar_fields_are_rejected(field, value, match):
    config = RunConfig(name="bad", datasets=(REFERENCE,), **{field: value})
    with pytest.raises(ConfigError, match=match):
        validate_run_config(config)


def test_unknown_survey_is_rejected():
    with pytest.raises(ConfigError, match="unknown survey"):
        validate_run_config(RunConfig(name="bad", surveys=("rgo", "soon")))


def test_missing_dataset_path_is_rejected():
    bad = DatasetRef("no_such_folder", "calibration", "sunspot")
    with pytest.raises(ConfigError, match="no such path"):
        validate_run_config(RunConfig(name="bad", datasets=(bad,)))


def test_unknown_role_is_rejected():
    bad = DatasetRef("silso", "training", "sunspot")
    with pytest.raises(ConfigError, match="unknown role"):
        validate_run_config(RunConfig(name="bad", datasets=(bad,)))


def test_yaml_round_trip(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "name": "round_trip",
                "seed": 7,
                "surveys": ["rgo"],
                "thresholds": [0.0, 30.0],
                "datasets": [{"path": "reference_groups_2026-10", "role": "calibration"}],
            }
        )
    )
    config = load_run_config(path)
    assert config.surveys == ("rgo",)
    assert config.thresholds == (0.0, 30.0)
    assert config.datasets[0].kind == "sunspot"


# --- lineage (ADR-002) ----------------------------------------------------


def test_lineage_names_every_input_and_its_role(tmp_path):
    config = load_run_config(CONFIGS / "step1_check_a.yaml")
    path = write_lineage(config, tmp_path)
    record = json.loads(path.read_text())

    assert record["run"] == "step1_check_a"
    assert record["seed"] == config.seed
    assert record["release"] == "v1.3_4"
    assert {i["path"] for i in record["inputs"]} == {d.path for d in config.datasets}
    for entry in record["inputs"]:
        assert entry["role"] in ("calibration", "comparison", "diagnostic")
        assert entry["sha256"] is not None


def test_lineage_states_the_firewall_explicitly():
    config = load_run_config(CONFIGS / "step1_check_a.yaml")
    record = lineage_record(config)
    assert record["firewall"]["variant_is_sunspot_only"] is True
    assert record["firewall"]["proxy_derived_calibration_inputs"] == []
    assert (
        "No isotope, aurora, proxy or SILSO input entered calibration"
        in (record["firewall"]["statement"])
    )


def test_lineage_admits_proxy_inputs_when_they_are_used():
    config = RunConfig(name="proxy", variant=PROXY_INFORMED, datasets=(REFERENCE, SILSO_CAL))
    record = lineage_record(config)
    assert record["firewall"]["proxy_derived_calibration_inputs"] == ["silso"]
    assert "Calibration used proxy-derived inputs" in record["firewall"]["statement"]


def test_lineage_records_file_checksums_not_just_names(tmp_path):
    """A lineage record that cannot be checked is not evidence."""
    dataset = DatasetRef("gn_database_v1.3/gn_v13_observations.parquet", "calibration")
    record = lineage_record(RunConfig(name="one", datasets=(dataset,)))
    entry = record["inputs"][0]
    assert entry["sha256"] and entry["sha256"] != "directory"
    assert len(entry["sha256"]) == 64


def test_two_runs_sharing_a_directory_keep_separate_lineage(tmp_path):
    """ADR-002 wants lineage per run, and several Step 2 scripts share a log
    directory. Writing ``lineage.json`` from each of them left only the last
    one's provenance, and two overlapping runs could tear the file."""
    from gn_next_gen.config import load_run_config, write_lineage

    config = load_run_config(CONFIGS / "step2_benchmark.yaml")
    first = write_lineage(config, tmp_path, extra={"script": "a"}, filename="lineage_a.json")
    second = write_lineage(config, tmp_path, extra={"script": "b"}, filename="lineage_b.json")

    assert first != second
    assert first.is_file() and second.is_file()
    assert json.loads(first.read_text())["script"] == "a"
    assert json.loads(second.read_text())["script"] == "b"


def test_lineage_still_defaults_to_the_step_1_filename(tmp_path):
    from gn_next_gen.config import load_run_config, write_lineage

    config = load_run_config(CONFIGS / "step1_check_a.yaml")
    assert write_lineage(config, tmp_path).name == "lineage.json"
