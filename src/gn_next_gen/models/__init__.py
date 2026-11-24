"""Generative components: the latent population, visibility, and detection.

``emissions`` is the detection layer Step 3 fits and Step 4 reuses --
``inference.forward_filter.count_emissions`` already promises that "Step 4
replaces this with per-observer soft-threshold emission tables; the filter does
not change".
"""

from gn_next_gen.models.dynamics import (
    birth_matrix,
    expected_lifetime_days,
    phi_from_lifetime,
    simulate_counts,
    simulate_paths,
    stationary_distribution,
    stationary_mean,
    survival_matrix,
    transition_matrix,
)
from gn_next_gen.models.emissions import (
    MAX_COUNT,
    DetectionGrid,
    build_detection_grid,
    detection_grid,
    emission_matrix,
    empirical_detection_table,
    log_emission_library,
    params_at_activity,
    reference_activity,
)

__all__ = [
    "MAX_COUNT",
    "DetectionGrid",
    "birth_matrix",
    "build_detection_grid",
    "detection_grid",
    "emission_matrix",
    "empirical_detection_table",
    "expected_lifetime_days",
    "log_emission_library",
    "params_at_activity",
    "phi_from_lifetime",
    "reference_activity",
    "simulate_counts",
    "simulate_paths",
    "stationary_distribution",
    "stationary_mean",
    "survival_matrix",
    "transition_matrix",
]
