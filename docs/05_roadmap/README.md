# Roadmap

Six steps from the assessment ([../01_assessment/assessment.md §5](../01_assessment/assessment.md)). Each step has a
spec here; a Claude Code session turns a spec into a plan (`/plan-step N` → `docs/plans/step-N-plan.md`), the PI
approves it, then `/implement-step N` builds it and `/verify-step N` checks it (`docs/results/step-N-verification.md`).

```
Step 1  Data foundation + viability checks
   │      (loaders, data issues, data refresh, birth-death/thinning check, area-distribution stationarity)
   ▼
Step 2  Blind benchmark  ──────────────────────────────┐ (scores every later method)
   │      (synthetic observers on the real graph; legacy baselines)
   ▼                                                    │
Step 3  Global network calibration (Approach 2)         │
   │      (concurrent fit, graph diagnostics, initializer)
   ▼                                                    │
Step 4  Joint latent-population model (Approach 1)  ◄───┘
   │      (exact forward filter, FFBS ensemble → sunspot-only series v0)
   ├──────────────► Step 5  Neural emission calibrator (only if Step 4 PPCs fail for some observers)
   └──────────────► Step 6  Proxy-informed variant (behind the firewall)
```

## Status

Update this table at the end of every `/implement-step` and `/verify-step`.

| Step | Spec | Plan | Status | Verification |
|---|---|---|---|---|
| 0 | Clean slate (this repository's initial content) | — | Done (2026-10-28) | — |
| 1 | [step-1-data-foundation-and-viability.md](step-1-data-foundation-and-viability.md) | — | Not started | — |
| 2 | [step-2-blind-benchmark.md](step-2-blind-benchmark.md) | — | Not started | — |
| 3 | [step-3-global-network-calibration.md](step-3-global-network-calibration.md) | — | Not started | — |
| 4 | [step-4-joint-latent-population-model.md](step-4-joint-latent-population-model.md) | — | Not started | — |
| 5 | [step-5-neural-emission-calibrator.md](step-5-neural-emission-calibrator.md) | — | Not started (conditional) | — |
| 6 | [step-6-proxy-informed-variant.md](step-6-proxy-informed-variant.md) | — | Not started | — |

Status values: Not started → Planned (plan approved) → In progress → Implemented → Verified (or Not needed, for Step 5).

## Conventions for all steps

- Read `CLAUDE.md` first; each spec lists its **Required reading**.
- Suggested package layout (refine in the Step 1 plan): `gn_next_gen.io` (loaders), `.reference` (area tables,
  thresholding, emission tables), `.graph` (overlap graph, diagnostics), `.benchmark` (generator, scenarios, scoring),
  `.baselines` (legacy-method ports), `.models` (network calibration, joint model), `.inference` (filters, samplers),
  `.proxies` (Step 6), `.viz`. Scripts that produce results go in `scripts/`; exploratory notebooks in `notebooks/` as
  py:percent text (jupytext) so they diff cleanly.
- Large outputs go to `outputs/` (git-ignored). Commit only code, configs, small summary tables/figures and results docs.
- Every result must be reproducible from a committed config + seed.
- Respect the hard rules in `CLAUDE.md` (immutable `data/raw` and `legacy/`, firewall, estimand, no SILSO in calibration).

## Methods-paper narrative (keep updated)

Existing methods are limiting cases of the joint model: ADF = hard threshold, binary emissions, missing at random;
backbone = single-path chained linking with the backbone treated as exact; probability-distribution matrices = pairwise
Dawid-Skene against a stand-in for truth. The benchmark shows what each generalization adds. Identifiability is reported
as first-class output: reference-equivalent scale, era-wide drift (prior sensitivity), effective resistance to the
reference, per-year prior→posterior contraction.

Running list of paper items (add as steps produce them):
- [ ] Data provenance table and known-issue resolutions (Step 1)
- [ ] Birth-death + thinning validation against reference statistics (Step 1)
- [ ] Benchmark design figure and baseline scores (Step 2)
- [ ] Overlap-graph diagnostics through time (Step 3)
- [ ] Benchmark comparison of all methods incl. coverage (Steps 3–4)
- [ ] GN posterior ensemble 1610–present, sunspot-only (Step 4)
- [ ] Maunder Minimum: blanket-statement and MNAR sensitivity, contraction map (Step 4)
- [ ] Proxy-informed variant and leave-one-proxy-type-out comparison (Step 6)
