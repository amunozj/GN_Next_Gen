# Step 5 — Neural emission calibrator (conditional)

## Trigger
Run only if Step 4 posterior-predictive checks fail for a material set of observers (or effects that are hard to write
as a likelihood matter, e.g. weather clustering, transcription errors). Otherwise mark the step **Not needed** in the
roadmap status table with the PPC evidence.

## Goal
Replace the parametric emission/observer model for the failing observers with a simulation-trained neural estimator
whose output plugs into the Step 4 likelihood, without giving up the shared latent Sun.

## Design (detail in the plan)
- Simulator: reference windows plus the Step 4 population model, with grand-minimum-like regimes by deleting whole
  groups; observer operator (threshold, splitting, drift, real calendar, MNAR zeros, blanket statements, extra effects).
- Estimator: neural likelihood or posterior estimation (NLE/NPE) over observer parameters from the raw daily record
  (sequence encoder with missing-day mask); amortized over observers. Hierarchical use: per-observer posteriors under a
  broad prior, then reweighting to the population prior.
- Diagnostics: simulation-based calibration, TARP, expected coverage; ensembles against overconfidence;
  out-of-distribution flags (especially Maunder-Minimum-era observers).
- Tooling: BayesFlow (JAX backend; API still changing), sbijax; sbi (PyTorch) for diagnostics if needed.

## Acceptance criteria (quantify in the plan)
Calibrated coverage on held-out simulations; improved PPCs for the targeted observers in Step 4; OOD flags reported;
benchmark scores do not degrade.

## Required reading
`CLAUDE.md`; assessment §4 (Approach 3); report section "Simulation-based inference upgrades ADF…" (all subsections),
ranked idea 9; note `sbi_and_scalable_inference.md`; Step 4 results doc (PPC failures).

## Out of scope
End-to-end SBI of the series (it discards cross-observer information).
