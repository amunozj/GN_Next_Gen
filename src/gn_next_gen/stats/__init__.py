"""Summary statistics and discrepancy measures."""

from gn_next_gen.stats.discrepancy import (
    TARGETS,
    CheckAScores,
    activity_bins,
    autocorrelation,
    block_bootstrap,
    count_pmf,
    dispersion_index,
    ks_distance,
    run_lengths,
    score_check_a,
    smooth_centered,
    total_variation,
    zero_fraction_by_bin,
)

__all__ = [
    "TARGETS",
    "CheckAScores",
    "activity_bins",
    "autocorrelation",
    "block_bootstrap",
    "count_pmf",
    "dispersion_index",
    "ks_distance",
    "run_lengths",
    "score_check_a",
    "smooth_centered",
    "total_variation",
    "zero_fraction_by_bin",
]
