# Step 2 verification: the blind benchmark of synthetic observers

- Plan: [`docs/plans/step-2-plan.md`](../plans/step-2-plan.md) (approved 2026-11-22, with ten in-session decisions)
- Results under verification: [`step-2-blind-benchmark.md`](step-2-blind-benchmark.md)
- Branch / commit verified: `step-2-blind-benchmark` @ `be05087`, working tree clean
- Environment: conda env **`gn-next-gen`**; Python 3.12.16, JAX 0.12.1 on
  `[CudaDevice(0..3)]` (4× A100), NumPy 2.5.5, pandas 3.1.1, SciPy 1.18.1, 64 cores
- Date: 2026-11-22

> **How this was run.** Every measurement below was re-taken from the committed
> code and the grid's own CSVs. Determinism was checked by regenerating into a
> scratch root; the A157 validation was re-run from scratch. Nothing here is
> copied from the implementation session's claims — and two of those claims
> turned out to be wrong (F1, F2).

## Verdict: **Pass with caveats**

The benchmark works, the grid is complete and reproducible, and its central
claims hold on re-measurement. Three acceptance criteria do not pass as written.

1. **A8 fails.** The criterion asks that the linear methods *amplify* strong
   cycles on the real overlap graph. They do not: HoSc98 −0.023 and SvSc16 −0.061
   against a required positive. The *families still separate* cleanly, and the
   same five methods do agree in sign on A157's own setup (A7), so what fails is
   the transfer of the published sign to the overlap graph — a result, not a bug.
2. **A7 passes in half.** Sign agreement 5 of 5; the exclusion limits do not
   reproduce (HoSc98 73 msd against 31), for a cause measured in the catalogue.
3. **A9 fails.** Three of the five named published series, not five.

Two criteria are **mis-specified** rather than failed — the numbers they name are
not reachable by the design the same plan describes (F3, F4). And two statements
in the results doc were **wrong and are corrected here** (F1, F2).

## Acceptance criteria

| Criterion | Measurement | Result | Pass? |
|---|---|---|---|
| Determinism: same config + seed → identical datasets | `generate_benchmark.py --check` regenerating S1–S5 into a scratch root, SHA-256 per file | **MATCH** on all five scenarios, every file | **Pass** |
| Scorer unit-tested on hand-made cases | `tests/test_scoring.py` | 15 passed. Nominal-coverage case within ±0.03 of 50/68/90/95; CRPS matches a 400,000-draw Monte-Carlo estimate to 2% | **Pass** |
| Every baseline scored on every scenario | `scores.csv` pivot | **459 cells**, 9 methods × 5 scenarios, **0 empty combinations** | **Pass** (the "42 cells" arithmetic is wrong — F3) |
| A port reproduces legacy thresholds | Dropped by PI decision 7 | Deviation recorded in the plan; recovered `runcsvs/` parameters are in `docs/03_legacy/adfssn.md` | **Superseded** by A7 |
| **A1** Truth sealed from fitting code | AST import audit over `src/`, plus `tests/test_benchmark_blindness.py` | **0** imports of `benchmark.truth` from `baselines`, `models` or `inference`; 10 tests pass, including one proving the audit catches a planted leak | **Pass** |
| **A2** Truth carries real areas *and* real lifetimes | Two-sample KS of the generated truth against the donor survey | area **KS 0.0065** (verbatim era) and **0.0082** (resequenced); `obs_duration` **0.0016** and **0.0083**; threshold 0.05. D22-colliding ids absent | **Pass** |
| **A3** Real calendars and the real graph | Per-station day sets, synthetic against real | **0 of 466** stations report a day they never had. 60 match exactly; 391 are strict subsets — MNAR omission and blanket conversion remove days by design. Graph: 423 stations, 16,858 edges, 14 components (largest 384), 20 bridges | **Pass on substance**, criterion mis-worded (F4) |
| **A4** Generator ≠ Step 4's model, four ways | The misspecification table, each switchable | All four present: disk-censored real lifetimes (median 5, max 15 days), donor-cycle emergence, 46 observers with breaks and 456 with drift, 23 copiers. Nine switches on `ObserverParams` | **Pass** |
| **A5** Interval coverage for every baseline, stratified by era | Coverage at 50/68/90/95 | Finite for **459 of 459** cells; 180 rows in `coverage.csv`; era stratification in this document | **Pass** |
| **A6** Hay sampling-only bias measured | Induced trend bias, bootstrapped | +0.0136 groups/century against a true 0.2865 (1610–); bootstrap sd falls 0.212 → 0.008 from 5 to 400 observers while the mean holds at +0.01 to +0.02 | **Pass**; range 5–400, not the planned 25–600 (only 467 observers exist) |
| **A7** Ports reproduce A157's published behaviour | `validate_against_a157.py`, re-run | **Sign: 5 of 5 agree.** HoSc98 −0.00157 and SvSc16 −0.00076 fall (99.2% and 90.8% of steps non-increasing); CEA17, UEA21, VHS24 at +0.0003 to +0.0006. **Exclusion limits: HoSc98 73 msd against their 31, CEA17 300 against 87** | **Partial** — sign passes, exclusion limits do not |
| **A8** A157's secular bias reproduces on the real graph | Cycle-amplitude slope, S2 | Linear **HoSc98 −0.023, SvSc16 −0.061** — the criterion requires positive. Non-parametric −0.218 to −0.412. Families separate strictly | **Fail as written** |
| **A9** Published GN series acquired | Folder, manifest, checksum test | 5 files present but only **3 of the 5 named** (Hoyt & Schatten, the SvSc16 backbone, an unversioned Usoskin-family series) plus Vaquero and Zolotova. `role: comparison_only`; `--check` MATCH; `test_data_manifest.py` 35 passed | **Fail** — 3 of 5 |
| **A10** Structural ensemble beside parametric uncertainty | Spread across 15 settings vs mean reported sd | 405 cells, 3 knobs × 5 settings × 3 worlds × 9 methods. **The ensemble is narrower than the reported uncertainty for every method** (e.g. Backbone 0.221 against 0.669) | **Pass** |
| **A12** Ten case studies shown individually | `case_study_observers.csv`, 3,500 rows | All ten present. Methods off by >2× in scale: `slow-drift` **7/7**, `clean-modern` 4, `copier` 4, `poor-instrument` 4, `leaf-node` 3, `break-undocumented` 2, `sparse-maunder` 1, `zero-omitter` 1 | **Pass** — all four required kinds show a failure |
| **A13** No long job run in session | `grid.log` | The grid was launched by the PI at 17:33:52 and ran 16 min detached. The session ran only S1 slices and 16-second validations | **Pass** |
| **A11** Figure-first results doc | Folder scan against the doc | **14 figures** f13–f26, both modes, largest **86 kB** (limit 400), all embedded as plain Markdown and all captioned `**Figure fNN**`; `tests/test_viz.py` 16 passed | **Pass** |

## Test suite

```
$ pytest -q
345 passed, 11 deselected in 58.35s

$ pytest -m slow -q
11 passed, 345 deselected in 402.53s (0:06:42)

$ ruff check . && ruff format --check .
All checks passed!
150 files already formatted
```

No failures, no skips beyond the marker split. The slow suite includes the
Step 2 additions: the verbatim-span fidelity test (RGO reproduced exactly over
24,219 days), UEA21's acuity recovery from active-day statistics, and the real
overlap graph on the full 1610–2026 record.

## The hard rules

| Rule | Check | Result |
|---|---|---|
| `legacy/` untouched | `git diff --name-status main -- legacy` | **empty** |
| `data/raw` additions only | `git diff --name-status main -- data/raw` | one modified file (`MANIFEST.csv`, 3 rows appended) and one new folder — exactly what ADR-006 allows |
| Manifest rows for new raw data | `grep`, `tests/test_data_manifest.py` | 3 rows; 35 tests pass |
| No SILSO in calibration | import audit of `src/` | SILSO appears only in `io/silso.py`, its re-export in `io/__init__.py`, and the firewall's own reject-list in `config/run_config.py`. **No module under `benchmark/`, `baselines/`, `models/` or `inference/` touches it** |
| Firewall (ADR-002) | `lineage_baselines.json` from the grid | `variant: sunspot_only`, and the explicit statement "No isotope, aurora, proxy or SILSO input entered calibration" |

Decision 3 is satisfied **by construction** rather than by substitution: the
published UEA21 derives acuity from active-day statistics and never needs a cycle
phase, so the SILSO-dependent scanning-window machinery of the 2017 ADFSSN engine
was simply not ported.

## Findings

### F1 — the results doc understated the structural ensemble (corrected)

§8 claimed the ensemble was "405 cells over four fast methods and one S2 world …
a slice". Measured from `structural.csv`: **nine methods, three worlds, three
knobs × five settings = 405 cells** — the full planned ensemble. The
implementation session had run a four-method slice by hand earlier and then
described the orchestrator's full run in the slice's terms.

Corrected in place, and the limitation "the structural ensemble is a slice" is
removed from §11.

### F2 — A10's actual answer was never given, and it sharpens §5

The criterion asks whether the ensemble is wider than the parametric
uncertainty. The results doc reported a magnitude and never answered the
question. Measured:

| method | structural spread | mean reported sd | wider? |
|---|---|---|---|
| HoSc98 | 0.192 | 0.645 | no |
| Backbone | 0.221 | 0.669 | no |
| CEA17 | 0.401 | 0.510 | no |
| VHS24 | 0.508 | 1.084 | no |

**For every method the ensemble is narrower than the interval already reported.**
That is a stronger statement than coverage alone: these intervals are not too
narrow because the methods ignored their own free choices — the choices move the
answer by less than the interval allows. Widening an interval to admit structural
choices would not fix the coverage. §8 now says so.

### F3 — "7 baselines × 6 scenarios = 42 cells" is arithmetically impossible

The plan's own design states that **S6 generates no worlds** — it perturbs S2 —
so there are five dataset scenarios, not six. The criterion's own number cannot
be produced by the design in the same document. The intent (every baseline scored
on every scenario) is met with 0 empty combinations across 459 cells. The
threshold is wrong, not the work.

### F4 — A3 as worded cannot pass, because MNAR is in the design

The criterion asks for an "exact match of observing-day sets for every
transplanted station". The generator deliberately omits spotless days
missing-not-at-random and converts some records to blanket statements, so a
reported set is a **subset** of the calendar for 391 of 466 stations. The
property that matters — nothing invented — holds exactly: **0 stations report a
day the real observer never had.** The criterion should have been written as
subset-plus-zero-violations.

### F5 — coverage gets *worse* toward the present

Not previously reported. Coverage at nominal 0.50, by era, on S2/S3:

| method | pre-1750 | 1750–1850 | 1850–1950 | post-1950 |
|---|---|---|---|---|
| HoSc98 | — | 0.218 | 0.170 | 0.082 |
| SvSc16 | — | 0.223 | 0.175 | 0.098 |
| Backbone | — | 0.185 | 0.155 | 0.098 |
| CEA17 | — | 0.100 | 0.040 | 0.016 |
| UEA21 | 0.034 | 0.045 | 0.010 | 0.000 |
| GRP | 0.028 | 0.020 | 0.005 | 0.006 |

Every method is **least** honest where the data are **best**. The inverse-variance
combination narrows the interval as observers accumulate faster than the error
actually falls, so overconfidence grows with network density. The dashes are the
reach result (§6 of the results doc): five methods produce nothing before 1750.

### F6 — A8's failure is informative and should not be smoothed away

On A157's own setup all five methods agree in sign (A7). On the real overlap
graph the linear group's positive slope does not survive: it sits at −0.02 to
−0.06 rather than above zero, while the non-parametric group stays at −0.22 to
−0.41. The *separation* between the families — the substance of A157's §4 —
reproduces; the sign of one family does not. The likely cause is that A157
compares each observer directly to RGO while here every observer is reached
through a chain, which is precisely the difference the benchmark was built to
measure. Recorded as a failed criterion rather than a reworded one.

## Figures

All 14 regenerate from the committed configs and the grid's CSVs. The four that
carry a verification claim are below.

![Nominal against empirical interval coverage at 50, 68, 90 and 95 percent; every curve far below the diagonal.](figures/step-2/f20_interval_coverage_light.png)

**Figure f20** — The step's central claim, re-measured: not one method's
intervals are honest. At nominal 0.50 the best of the seven delivers 0.159 and
the worst 0.009. F5 adds that this gets worse toward the present, which the
pooled figure cannot show.

![Slope of estimated-minus-true cycle amplitude against true amplitude, per method, with error bars.](figures/step-2/f19_cycle_amplitude_light.png)

**Figure f19** — Criterion **A8**, which fails. The two families separate
cleanly, but the linear group sits at zero rather than above it, where A157
places it. The figure's own caption states this rather than claiming the
published result.

![Mean and RMS difference against acuity threshold for five methods on the A157 setup.](figures/step-2/f24_a157_validation_light.png)

**Figure f24** — Criterion **A7**, re-run from scratch during verification. All
five methods agree in sign with the published Fig. 3b. The exclusion limits do
not reproduce, and `a157_catalogue.csv` carries the measured reason.

![Bars spanning each method's yearly bias across its own free graph choices.](figures/step-2/f22_structural_ensemble_light.png)

**Figure f22** — Criterion **A10**, corrected by F1 and F2: the full 405-cell
ensemble, and the ensemble is **narrower** than the uncertainty the methods
already report.

CSVs behind every number: `outputs/step2/scores/{scores,coverage,structural,case_study_observers}.csv`,
`outputs/step2/a157/{a157_validation,a157_verdict,a157_catalogue}.csv`,
`outputs/step2/sampling_only/sampling_only_{trend,bootstrap,summary}.csv`.

## Deviations from the plan

The plan records eight. Verification confirms all eight and adds the corrections
in F1–F4. Beyond those:

- **A6's bootstrap spans 5–400 observers, not 25–600.** The network has 467
  stations, so 600 is unreachable; the implementation also added sizes below the
  planned floor, which is where the interesting widening is.
- **One anchoring, not two.** Every scenario uses ADR-009's secondary anchor
  (RGO 15–19). The Debrecen 21–23 anchoring is supported and configured per
  scenario but was not run, so the anchor-sensitivity result ADR-003 asks for is
  still owed.

## Findings that affect later steps

1. **The bar and the floor, for Steps 3–4.** Yearly cRMSE **0.312** (HoSc98) and
   a 50% coverage of **0.159** (Backbone) are the best of the seven. A random
   segmentation scores **0.749**; four published methods are below it. A new
   method that does not clear the floor has demonstrated nothing.
2. **Coverage is the open problem, and F5 says where it is worst.** Overconfidence
   grows with network density. A Step 3 composite-likelihood correction should be
   validated per era, not pooled, or it will be tuned on the easy end.
3. **Five of seven methods cannot reach before 1774.** This is the gap the joint
   model exists to close and it is now measured. Step 3's effective-resistance
   diagnostics should report, per era, how much of the record a chained method
   can reach at all — the benchmark's `graph.resistance_to_set` already does this.
4. **A8's failure belongs in the methods paper.** That the published secular-bias
   sign does not survive the move from pairwise-to-RGO onto a real overlap graph
   is a result about daisy-chaining, which A157 explicitly did not evaluate.
5. **No ADR change is needed.** Nothing here contradicts an accepted decision.
   A9's shortfall is recorded in `refresh_todo.md`; the anchor-sensitivity gap is
   ADR-003's existing requirement, now explicitly owed by Steps 3–4.
6. **Two criteria should be reworded if this spec is ever reused** (F3, F4). The
   roadmap spec for Step 2 is unchanged; the corrections live here rather than
   being edited silently into it.
