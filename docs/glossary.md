# Glossary

| Term | Meaning here |
|---|---|
| GN | Group (sunspot) number: number of sunspot groups on the visible disk; the series this project reconstructs |
| SN / ISN | (International) sunspot number, SILSO V2.0; a composite of spots and groups. Not used for calibration (ADR-003) |
| Observer / station | An observing record in the GN database, keyed by `STATION` (1–775 in v1.3; 740 carry data). One observer per station, but **not one record per station-day**: 217 `(date, station)` keys are duplicated in v1.3, 95 with conflicting counts (D16) |
| Reference survey | Five per-survey catalogues of individual group areas, one row per group observation: Schwabe (1825–1867), Spörer (1861–1894), RGO (1874–1976), Debrecen DPD (1974–2018), Kislovodsk KMAS (1954–2026). `data/raw/reference_groups_2026-10/`; ADR-009 |
| Anchored epoch | The reference segment whose detection parameters are held fixed, so GN is counted in its units (ADR-003). Two co-equal anchors: **Debrecen cycles 21–23** (primary) and **RGO cycles 15–19** (secondary), each whole cycles covered by one survey, disjoint in cycle number (ADR-009) |
| `day_status` | Per (survey, day) in the reference: 0 = no observation, 1 = observed spotless, 2 = observed active. Replaces the legacy `AREA1` NaN/0 convention |
| `group_obs_kind` | Per group in the reference: 0 = placeholder row, 1 = single-observation group, 2 = multi-observation group. Distinct from `day_status`; both are called `obs_type` upstream (D10) |
| MSH | Millionths of the solar hemisphere (area unit) |
| Acuity threshold | Smallest group area (MSH) an observer would see; "soft" when detection probability rises smoothly with area |
| Splitting / merging | Observer conventions that report more / fewer groups than the reference for the same spots |
| k-factor | Multiplicative ratio between an observer's and a reference's counts (classic calibration; activity-dependent) |
| ADF | Active day fraction: fraction of observed days in a "month" with ≥1 group. ADF calibration matches its distribution to thresholded reference windows (legacy ADFSSN) |
| QDF | Quiet day fraction (1 − ADF-type quantity used at high activity in the legacy DTh mode) |
| EMD | Earth Mover's (Wasserstein) distance between histograms (legacy fingerprint distance) |
| Backbone | Calibration through a chain of long-running reference observers (Svalgaard & Schatten 2016) |
| Daisy chain | Pairwise calibration passed along a chain of overlapping observers |
| Overlap graph | Nodes = observers, edges = pairs with near-simultaneous observations |
| Effective resistance | Graph quantity equal to the variance of a concurrently estimated scale between two nodes (research derivation); grows with chain length, shrinks with parallel paths |
| Reference-equivalent count | The count the fixed reference observer would have made (the estimand; ADR-003) |
| Immigration-death / INAR | Latent daily count dynamics: survivors ~ Binomial(N, φ), births ~ Poisson(λ) |
| Emission | P(observed count \| latent count, observer parameters) |
| MNAR | Missing not at random; here: spotless days less likely to be recorded |
| Blanket statement | Qualitative report such as "no spots seen in year X", with unknown observing dates |
| FFBS | Forward-filter backward-sample: exact posterior draws of a discrete latent chain |
| Blocked Gibbs | Alternating exact FFBS for the latent counts with gradient-based updates of observer parameters |
| SMC | Sequential Monte Carlo (tempered SMC gives marginal likelihoods for model comparison) |
| SBI / NPE / NLE | Simulation-based inference; neural posterior / likelihood estimation |
| SBC / TARP | Simulation-based calibration; Tests of Accuracy with Random Points (posterior coverage diagnostics) |
| PPC | Posterior-predictive check |
| MM | Maunder Minimum, ~1645–1715 |
| PSM | Proxy system model: forward model from the latent state to a proxy observation |
| Firewall | Data-lineage rule keeping proxy information out of the sunspot-only variant (ADR-002) |
| [nv] | In research docs: reference cited from memory, not re-verified |
