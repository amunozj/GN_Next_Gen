# Step 6 — Proxy-informed variant (behind the firewall)

## Goal
A second released series in which independent proxies inform the latent activity, mainly where sunspot data are sparse
(Maunder Minimum, early 1600s, gaps), with documented data lineage so the sunspot-only series stays independent.

## Design (detail in the plan)
- Differentiable proxy system models from λ(t) (or open solar flux derived from it): open flux → heliospheric modulation
  → ¹⁴C production → carbon-cycle box model → annual Δ¹⁴C (e.g. the JAX/diffrax `ticktack` package); ¹⁰Be with
  transport/deposition noise; aurora counts as Poisson with a rate depending on activity and observing effort;
  optionally geomagnetic indices for the modern era.
- Each proxy has its own noise model; temporal resolution mismatch (annual vs daily) handled by aggregation of the latent.
- Firewall (ADR-002): proxy terms switchable; no proxy information in the sunspot-only variant's priors or
  hyperparameters; lineage recorded per run.
- Outputs: proxy-informed ensemble; leave-one-proxy-type-out siblings; per-year prior→posterior contraction maps for both
  variants; consistency analysis where sunspot and proxy information overlap.

## Inputs to acquire (Step 6 plan)
Annual ¹⁴C (Brehm et al. 2021) and other ¹⁴C records; ¹⁰Be records; aurora catalogues; production/transport model
parameters. Add `MANIFEST.csv` rows.

## Acceptance criteria (quantify in the plan)
Reproducible variants; firewall verified by a lineage check (test) — the raw snapshots already
carry explicit lineage statements in their `provenance.json` (`data_lineage`, naming what is
excluded: no isotope, aurora, proxy or SILSO input), so the test should extend that pattern to
run lineage rather than invent one; contraction maps produced; leave-one-out results
reported; Maunder-Minimum level and cyclicity compared between variants with uncertainties.

## Precedents
Usoskin et al. 2021 (A&A 649, A141) physics chain with Monte Carlo errors; Schanner, Nilsson & Muscheler 2025 (Bayesian
GP solar modulation in PyMC/NumPyro on JAX, no sunspots). No published joint Bayesian model of GN with ¹⁴C, ¹⁰Be and
aurorae was found, so this variant is novel.

## Required reading
`CLAUDE.md`; ADR-002; assessment §2 (proxies); report sections "Isotopes and aurorae…", "Circularity firewalls…",
"Integrated models join counts…", ranked idea 10; note `paleoclimate_historical_evidence.md`; Step 4 results doc.

## Out of scope
Changing the sunspot-only series.
