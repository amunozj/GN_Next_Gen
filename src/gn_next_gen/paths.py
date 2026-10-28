"""Repository-relative paths.

Resolved from this file's location, so they are valid for the editable install created by
the conda environment files (``pip install -e .``).
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_INTERIM = REPO_ROOT / "data" / "interim"
MANIFEST = DATA_RAW / "MANIFEST.csv"

CONFIGS = REPO_ROOT / "configs"
OUTPUTS = REPO_ROOT / "outputs"
SCRIPTS = REPO_ROOT / "scripts"
LEGACY = REPO_ROOT / "legacy"
DOCS = REPO_ROOT / "docs"

# --- Reference survey -------------------------------------------------------
# Current reference: per-group, per-day catalogues for the core five surveys
# (Schwabe, Spoerer, RGO, Debrecen, KMAS), 1825-2026.  See ADR-009.
REFERENCE_GROUPS = DATA_RAW / "reference_groups_2026-10"
REFERENCE_GROUP_DAY = REFERENCE_GROUPS / "group_day_observations.parquet"
REFERENCE_GROUP_SINGLE = REFERENCE_GROUPS / "group_single_entry.parquet"
REFERENCE_DAY_STATUS = REFERENCE_GROUPS / "day_status.parquet"
REFERENCE_SURVEY_CALIBRATION = REFERENCE_GROUPS / "survey_calibration.csv"
REFERENCE_CYCLE_COVERAGE = REFERENCE_GROUPS / "survey_cycle_coverage.csv"

# Superseded day x AREA1..AREA26 matrices, kept for the legacy baselines (Step 2)
# and for the D3 characterization.  Do not use as the reference.
REFERENCE_AREAS_LEGACY = DATA_RAW / "reference_areas"

# --- Group-number observation database --------------------------------------
# Current observation database: v1.3 revised collection, 1610-2010.
GN_DATABASE_V13 = DATA_RAW / "gn_database_v1.3"
GN_V13_XLSX = GN_DATABASE_V13 / "Revised collection_v1.3_3.xlsx"
GN_V13_OBSERVATIONS = GN_DATABASE_V13 / "gn_v13_observations.parquet"
GN_V13_OBSERVERS = GN_DATABASE_V13 / "gn_v13_observers.parquet"
GN_V13_CHANGELOG = GN_DATABASE_V13 / "gn_v13_changelog.csv"

# Superseded, kept so Step 2 can reproduce the legacy baselines against the
# database the legacy code actually used.
GN_DATABASE_V122 = DATA_RAW / "gn_database_v1.22"

# Daily group counts from KMAS and SWPC, extending the observation record past
# v1.3's 2010 end.  KMAS is dual-use: see the folder README's single-use rule.
GN_OBSERVER_EXTENSION = DATA_RAW / "gn_observer_extension_2026-10"
GN_EXTENSION_COUNTS = GN_OBSERVER_EXTENSION / "observer_counts.parquet"
GN_EXTENSION_STATIONS = GN_OBSERVER_EXTENSION / "station_assignment.csv"

SILSO = DATA_RAW / "silso"
