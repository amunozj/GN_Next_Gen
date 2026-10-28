"""Verify that every raw data file matches its row in data/raw/MANIFEST.csv.

Fails if a raw file was edited, truncated, or had its line endings converted on checkout.
"""

import csv
import hashlib

import pytest

from gn_next_gen.paths import DATA_RAW, MANIFEST


def _manifest_rows():
    with MANIFEST.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


ROWS = _manifest_rows()


def test_manifest_is_not_empty():
    assert ROWS, "MANIFEST.csv has no rows"


@pytest.mark.parametrize("row", ROWS, ids=[r["path"] for r in ROWS])
def test_raw_file_matches_manifest(row):
    path = DATA_RAW / row["path"]
    assert path.is_file(), f"missing raw file: {row['path']}"
    assert path.stat().st_size == int(row["bytes"]), f"size changed: {row['path']}"
    assert _sha256(path) == row["sha256"], f"checksum changed: {row['path']}"


def test_every_raw_file_is_listed():
    listed = {r["path"] for r in ROWS}
    present = {
        p.relative_to(DATA_RAW).as_posix()
        for p in DATA_RAW.rglob("*")
        if p.is_file() and p.name != "MANIFEST.csv"
    }
    assert present <= listed, f"raw files missing from MANIFEST.csv: {sorted(present - listed)}"
