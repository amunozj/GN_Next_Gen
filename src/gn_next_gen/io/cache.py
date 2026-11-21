"""Checksum-keyed caching of cleaned tables into ``data/interim/``.

A cached file is keyed by the SHA-256 of every raw input it was built from --
taken from ``data/raw/MANIFEST.csv``, not recomputed -- plus the loader options.
So a cache built from a superseded release, or with a different duplicate rule,
can never be mistaken for the one asked for: the key simply does not match and
the table is rebuilt.  ``data/interim/`` is git-ignored and disposable.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from gn_next_gen.paths import DATA_INTERIM, DATA_RAW, MANIFEST

_MANIFEST_CACHE: dict[str, str] | None = None


def manifest_checksums() -> dict[str, str]:
    """``{relative path: sha256}`` for every row of the raw-data manifest."""
    global _MANIFEST_CACHE
    if _MANIFEST_CACHE is None:
        with MANIFEST.open(newline="") as handle:
            _MANIFEST_CACHE = {row["path"]: row["sha256"] for row in csv.DictReader(handle)}
    return _MANIFEST_CACHE


def manifest_checksum(path: Path) -> str:
    """The manifest's SHA-256 for one raw file.

    Raises
    ------
    KeyError
        If the file is not in the manifest.  Every raw input must be, so this is a
        real error rather than a reason to fall back to hashing the bytes: an
        unmanifested file has no provenance (ADR-006).
    """
    rel = str(Path(path).resolve().relative_to(DATA_RAW.resolve()))
    checksums = manifest_checksums()
    if rel not in checksums:
        raise KeyError(f"{rel} is not in {MANIFEST}; raw inputs must carry a manifest row")
    return checksums[rel]


def cache_key(inputs: list[Path], options: dict) -> str:
    """A short, stable digest of the inputs' checksums and the loader options."""
    payload = {
        "inputs": {str(Path(p).name): manifest_checksum(p) for p in sorted(inputs)},
        "options": options,
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def cached_frame(
    name: str,
    inputs: list[Path],
    options: dict,
    build: Callable[[], pd.DataFrame],
    *,
    use_cache: bool = True,
    force: bool = False,
) -> pd.DataFrame:
    """Return ``build()``'s result, reading or writing ``data/interim/`` as allowed."""
    if not use_cache:
        return build()

    path = DATA_INTERIM / f"{name}__{cache_key(inputs, options)}.parquet"
    if path.is_file() and not force:
        return pd.read_parquet(path)

    frame = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return frame
