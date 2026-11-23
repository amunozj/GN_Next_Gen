"""Per-run data lineage (ADR-002).

Every run writes a ``lineage.json`` naming each raw input, its manifest SHA-256
and the role it played, plus an explicit statement of whether any proxy-derived
information entered.  The firewall is a claim about a run; this is the evidence.
"""

from __future__ import annotations

import datetime as dt
import json
import platform
from pathlib import Path

from gn_next_gen.config.run_config import FIREWALLED_KINDS, SUNSPOT_ONLY, RunConfig
from gn_next_gen.io.cache import manifest_checksum


def lineage_record(config: RunConfig, *, extra: dict | None = None) -> dict:
    """Build the lineage record for a run."""
    inputs = []
    for dataset in config.datasets:
        entry = {"path": dataset.path, "role": dataset.role, "kind": dataset.kind}
        resolved = dataset.resolved
        if resolved.is_file():
            try:
                entry["sha256"] = manifest_checksum(resolved)
            except KeyError:
                entry["sha256"] = None
        else:
            entry["sha256"] = "directory"
        inputs.append(entry)

    calibration_kinds = {d.kind for d in config.datasets_in_role("calibration")}
    firewalled = sorted(calibration_kinds & set(FIREWALLED_KINDS))
    return {
        "run": config.name,
        "variant": config.variant,
        "seed": config.seed,
        "written_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "release": config.release,
        "kmas_role": config.kmas_role,
        "duplicate_rule": config.duplicate_rule,
        "surveys": list(config.surveys),
        "thresholds": list(config.thresholds),
        "inputs": inputs,
        "firewall": {
            "variant_is_sunspot_only": config.variant == SUNSPOT_ONLY,
            "proxy_derived_calibration_inputs": firewalled,
            "statement": (
                "No isotope, aurora, proxy or SILSO input entered calibration."
                if not firewalled
                else f"Calibration used proxy-derived inputs: {firewalled}."
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        **(extra or {}),
    }


def write_lineage(
    config: RunConfig,
    out_dir: str | Path,
    *,
    extra: dict | None = None,
    filename: str = "lineage.json",
) -> Path:
    """Write a lineage record into a run's output directory and return its path.

    ``filename`` exists because several scripts can share one output directory.
    ADR-002 wants lineage **per run**, so two runs writing ``lineage.json`` side
    by side would leave only the last one's provenance -- and if they overlap in
    time, a torn file. Scripts that share a directory pass distinct names.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / filename
    path.write_text(json.dumps(lineage_record(config, extra=extra), indent=2) + "\n")
    return path
