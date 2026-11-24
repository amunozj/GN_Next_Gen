"""Validators that refuse a run rather than letting it produce a tainted posterior."""

from __future__ import annotations

from gn_next_gen.config.run_config import (
    CIRCULAR_KINDS,
    FIREWALLED_KINDS,
    ROLES,
    SUNSPOT_ONLY,
    VARIANTS,
    RunConfig,
)
from gn_next_gen.io.observations import DUPLICATE_RULES
from gn_next_gen.io.reference import CORE_FIVE

KMAS_ROLES = ("reference", "observer_counts")

#: Paths that are KMAS in each of its two guises.  The single-use rule is about
#: KMAS's *observations*, which both of these carry.
KMAS_REFERENCE_PATH = "reference_groups_2026-10"
KMAS_COUNTS_PATH = "gn_observer_extension_2026-10"


class ConfigError(ValueError):
    """A run configuration that must not be executed."""


def validate_run_config(config: RunConfig) -> RunConfig:
    """Check every rule a run can break, and raise on the first one it does."""
    if config.variant not in VARIANTS:
        raise ConfigError(f"unknown variant {config.variant!r}; expected one of {VARIANTS}")
    if config.kmas_role not in KMAS_ROLES:
        raise ConfigError(
            f"kmas_role must be one of {KMAS_ROLES}, got {config.kmas_role!r} "
            "(ADR-009: KMAS is reference areas or observer counts, never both)"
        )
    if config.duplicate_rule not in DUPLICATE_RULES:
        raise ConfigError(
            f"duplicate_rule must be one of {DUPLICATE_RULES}, got {config.duplicate_rule!r}"
        )
    unknown_surveys = set(config.surveys) - set(CORE_FIVE)
    if unknown_surveys:
        raise ConfigError(f"unknown survey(s) {sorted(unknown_surveys)}; expected {CORE_FIVE}")

    for dataset in config.datasets:
        if dataset.role not in ROLES:
            raise ConfigError(
                f"{dataset.path}: unknown role {dataset.role!r}; expected one of {ROLES}"
            )
        if not dataset.resolved.exists():
            raise ConfigError(f"{dataset.path}: no such path under data/raw")

    _check_firewall(config)
    _check_no_circular_calibration(config)
    _check_kmas_single_use(config)
    return config


def _check_firewall(config: RunConfig) -> None:
    """ADR-002: the sunspot-only variant calibrates on sunspot data alone."""
    if config.variant != SUNSPOT_ONLY:
        return
    offenders = [
        d.path for d in config.datasets_in_role("calibration") if d.kind in FIREWALLED_KINDS
    ]
    if offenders:
        raise ConfigError(
            f"firewall (ADR-002): the {SUNSPOT_ONLY} variant may not calibrate on "
            f"{sorted(offenders)}. SILSO, isotopes, aurorae and other proxies may be "
            "carried with role 'comparison' or 'diagnostic', never 'calibration'."
        )


def _check_no_circular_calibration(config: RunConfig) -> None:
    """ADR-003: a published GN series is a comparison, never a calibration input.

    Unlike the firewall this applies to **both** variants. The proxy-informed
    variant may calibrate on isotopes; neither variant may calibrate on somebody
    else's reconstruction of the same observers, which would make the posterior a
    measurement of that author's daisy chain rather than of the record.
    """
    offenders = [d.path for d in config.datasets_in_role("calibration") if d.kind in CIRCULAR_KINDS]
    if offenders:
        raise ConfigError(
            f"circular calibration (ADR-003): no variant may calibrate on {sorted(offenders)}. "
            "A published group-number series is a recalibration of the same observers this "
            "model recalibrates; carry it with role 'comparison' or 'diagnostic'."
        )


def _check_kmas_single_use(config: RunConfig) -> None:
    """ADR-009: KMAS's observations may enter the likelihood once, in one guise."""
    calibration = config.datasets_in_role("calibration")
    uses_reference = any(KMAS_REFERENCE_PATH in d.path for d in calibration)
    uses_counts = any(KMAS_COUNTS_PATH in d.path for d in calibration)

    kmas_in_reference = uses_reference and "kmas" in config.surveys
    kmas_in_counts = uses_counts and config.kmas_role == "observer_counts"

    if kmas_in_reference and kmas_in_counts:
        raise ConfigError(
            "KMAS single-use rule (ADR-009): this run calibrates on KMAS as a "
            "reference survey *and* as observer counts, which puts the same "
            "observations into the likelihood twice and makes the posterior "
            "overconfident. Either drop 'kmas' from surveys or set "
            "kmas_role: reference."
        )
    if config.kmas_role == "observer_counts" and "kmas" in config.surveys and uses_reference:
        raise ConfigError(
            "KMAS single-use rule (ADR-009): kmas_role is 'observer_counts' but "
            "'kmas' is still in surveys while the reference snapshot is a "
            "calibration input. Remove it from surveys to make the role explicit."
        )
