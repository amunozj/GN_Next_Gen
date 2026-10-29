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

### Out-of-band work already landed

| Work | Plan | Status | Results |
|---|---|---|---|
| Raw-data renovation (2026-10) | [raw-data-renovation-plan.md](../plans/raw-data-renovation-plan.md) | Done (2026-10-29) | [raw-data-renovation.md](../results/raw-data-renovation.md) |

**Step 1 starts from renovated data, not from the clean-slate snapshot.** The reference survey is now
five per-survey, per-group catalogues covering 1825–2026 with group lifetimes
(`data/raw/reference_groups_2026-10/`), the observation database is v1.3 (1610–2010), and
`gn_observer_extension_2026-10/` carries daily counts to 2026. [ADR-009](../06_decisions/ADR-009-reference-survey-and-anchored-epochs.md)
settles the reference and the two anchored epochs that ADR-003 had deferred to Step 1, so the spec's
first two open questions are already answered.

Four issues are open and bite in Step 1 — **D16** (do not key on `(date, station)`), **D17** (size K
from a robust quantile, not `max(GROUPS)`), **D22** (exclude the colliding KMAS groups from lifetime
work) and the **KMAS single-use rule** (documentation-only today). All four are in `CLAUDE.md`'s
data-gotchas section and in the Step 1 spec.

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
- [x] Data provenance table and known-issue resolutions — mostly delivered by the 2026-10 renovation
      ([results](../results/raw-data-renovation.md), ADR-009, `data/raw/MANIFEST.csv`, D1–D22 statuses).
      Still to add in Step 1: resolutions for D16, D17, D22 and the KMAS single-use rule
- [ ] Anchor-sensitivity comparison: the series under both ADR-009 anchorings (Steps 3–4)
- [ ] Birth-death + thinning validation against reference statistics (Step 1)
- [ ] Benchmark design figure and baseline scores (Step 2)
- [ ] Overlap-graph diagnostics through time (Step 3)
- [ ] Benchmark comparison of all methods incl. coverage (Steps 3–4)
- [ ] GN posterior ensemble 1610–present, sunspot-only (Step 4)
- [ ] Maunder Minimum: blanket-statement and MNAR sensitivity, contraction map (Step 4)
- [ ] Proxy-informed variant and leave-one-proxy-type-out comparison (Step 6)
