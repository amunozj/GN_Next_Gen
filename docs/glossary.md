# Glossary

| Term | Meaning here |
|---|---|
| GN | Group (sunspot) number: number of sunspot groups on the visible disk; the series this project reconstructs |
| SN / ISN | (International) sunspot number, SILSO V2.0; a composite of spots and groups. Not used for calibration (ADR-003) |
| Observer / station | One record in the GN database; `STATION` code; currently one observer per station |
| Reference survey | RGO (1874–1976), Debrecen DPD, Kislovodsk (KMAS): daily lists of individual group areas |
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
