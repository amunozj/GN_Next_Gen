# ADR-007: Benchmark before claims; present existing methods as limiting cases

- Status: Accepted (2026-10-28, from the assessment)
- Decision: No method (new or legacy) is claimed better than another without the Step 2 blind benchmark: synthetic
  observers built from the reference on the real overlap graph and calendars, fitted with a model different from the
  generator, scored at several time scales and for uncertainty coverage. The methods paper frames ADF (hard threshold,
  binary emissions, missing at random), backbone (single-path chained linking) and probability-distribution matrices
  (pairwise Dawid-Skene) as limiting cases of the joint model, and shows on the benchmark what each generalization adds.
- Consequences: Step 2 precedes Steps 3–4; benchmark code and seeds are versioned; results docs report coverage, not only
  point errors. The comparison must also be positioned against Chatzistergos et al. (2025, A&A 699, A157).
