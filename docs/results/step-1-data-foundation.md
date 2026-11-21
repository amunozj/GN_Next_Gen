# Results: Step 1 — data foundation and viability checks

- Plan: [`docs/plans/step-1-plan.md`](../plans/step-1-plan.md) (Approved 2026-11-21)
- Spec: [`docs/05_roadmap/step-1-data-foundation-and-viability.md`](../05_roadmap/step-1-data-foundation-and-viability.md)
- Branch: `step-1-data-foundation`
- Environment: conda env **`gn-next-gen`**, created from `environment.yml` on this machine
  (Python 3.12, JAX 0.12.1 with CUDA on 4× A100, NumPyro 0.22, pandas 3.1.1)
- Decisions taken in session by the PI, 2026-11-21: Step 1 uses RGO/Debrecen/KMAS only;
  Check A's thresholds are advisory on this pass; v1.3_4 supersedes v1.3_3 in place
  ([ADR-010](../06_decisions/ADR-010-in-place-release-supersession.md)); `duplicate_rule`
  defaults to `max`; **K = 41**

> **Every number below is produced by a committed script and read from a committed CSV.**
> Verification in this step is figure-first: each claim names the figure that shows it.

## Summary

| Question | Answer |
|---|---|
| Is the area distribution stationary across the reference era? | **Yes**, across the anchor-era epochs — percentile spread 0.034–0.083 dex against a 0.1 dex target. **Early RGO (cycles 12–14) is the exception**, independently corroborating D5 |
| Is there two-population structure in areas? | **Yes, decisively** — ΔBIC +965 to +4,268 for two lognormal components in every epoch, with stable component means |
| Does an immigration–death chain reproduce the reference's daily statistics? | *(see Check A below)* |
| What should Step 4 use? | K = 41; φ is a **visible-residence** parameter, not an intrinsic lifetime; lifetimes need an explicit rotation kernel |
| Is the forward filter affordable at full length? | **Yes, on CPU**: 0.667 s per evaluation at T = 152,000, K = 41. **Not on GPU**: 6.95 s, ten times slower |

## 1. The data foundation

### 1.1 A new release of the observation database

The PI supplied `Revised collection_v1.3_4.xlsx` on 2026-11-21. It fixes most of what this
project reported upstream in [`gn-v13-feedback-report.md`](gn-v13-feedback-report.md):

| | v1.3_3 | **v1.3_4** |
|---|---|---|
| Data rows | 1,156,673 | 1,156,368 |
| Rows with `STATION > 0` | 1,100,144 | 1,099,986 |
| Stations with data | 740 | 740 |
| Observers listed | 775 | 776 |
| **D16** duplicate `(date, station)` keys | 217 (95 conflicting) | **3 (1 conflicting)** |
| **D17** `max(GROUPS)` | 93 | **58** |
| **D18** string `'NA'` sentinel | 33 observers | **none** |
| **D19** sheet flag column | `OBSER` | **`OBSERVER`** |
| **D20** changed-in-v1.3 | green cell fill only | **explicit column** |
| Row order | unsorted | **date-sorted** |

Observer-day diff: **144 new, 78 removed, 351 with changed counts** — see `f02`. Net rows fall
because removing the ~428 duplicate rows outweighs the additions, so this is a
correction-and-deduplication release, not an extension. Coverage still stops at 2010-12-31.

**Where the corrections land is the striking part.** 99% of the changed counts, 100% of the new
observer-days and 96% of the removals fall in **1715–1730**, concentrated in **1719–1721**
(143, 117 and 61 changed counts respectively). That is the end of the Maunder Minimum and the
start of the recovery — precisely the era this project's grand-minimum question turns on, and
precisely where ADR-008 says the series must be calibrated rather than merely asserted. Anyone
comparing a Maunder-era result against one computed from v1.3_3 is comparing different data.

Landing it in place contradicted ADR-006, so it carries **ADR-010** rather than a silent
deviation. Both workbooks are retained byte-exact; `convert_gn_database_v13.py --release v1.3_3`
regenerates the superseded tables into `data/interim/` and refuses to write them into
`data/raw/`; and `test_v133_still_reproduces` (slow) checks that they come back with their own
pinned figures, so every number in the renovation report and the feedback report stays
reproducible.

Two things worth knowing that the diff turned up:

- the derived observations parquet is **40% smaller** than the file it replaced, purely because
  `_4`'s rows are date-sorted and sorted `ORDINAL`/`FRACYEAR` columns compress far better;
- `_4`'s explicit changed-in-v1.3 column marks **100** observers against the green fill's **99**,
  disagreeing on exactly one: **station 36 (MOUTON, G., LYON)**, a placeholder with no
  observations. The converter prefers the column and prints the disagreement. Worth passing to
  the v1.3 authors as a loose end rather than a defect.

### 1.2 Loaders

`gn_next_gen.io` reads every raw input in long format — nothing densifies a day × observer
matrix. Cleaned tables cache to git-ignored `data/interim/` under a key built from the inputs'
**manifest checksums** plus the loader options, so a cache built from another release or another
duplicate rule cannot be read by mistake.

Coverage and sentinels are checked against `MANIFEST.csv` itself rather than against constants in
the test file (`f01`). The record's first observation is Harriot on **1610-12-18**; the manifest's
1610-01-01 is the table's calendar start, which is a no-observation sentinel row.

### 1.3 The rules that are now code, not prose

| Rule | Where it was | Where it is now |
|---|---|---|
| **Firewall** (ADR-002) — no SILSO/isotope/aurora/proxy in sunspot-only calibration | documentation | `config.validate_run_config` refuses the run; every run writes `lineage.json` with an explicit statement |
| **KMAS single-use** (ADR-009) — reference areas *or* observer counts | documentation | the config validator *and*, independently, `io.load_observations`, which withholds KMAS counts whenever `kmas_role` is `reference` |
| **D16** duplicate keys | "choose a rule in Step 1" | `io.resolve_duplicate_keys`, four rules, `max` by default |
| **D17** state-space sizing | "use a robust quantile" | a stated rule with both releases' numbers, pinned by a test (`f03`) |
| **D22** KMAS group-id collisions | "exclude them from lifetime work" | dropped by default in group-keyed tables, kept in daily counts, both tested |

### 1.4 A new issue: D23

**Schwabe and Spörer record no spotless days at all**, and Spörer tracks no group across days.
Measured from the snapshot and shown in `f05`:

| survey | no observation | **spotless** | active | groups | obs_duration |
|---|---|---|---|---|---|
| schwabe | 7,377 | **0** | 8,328 | 6,903 | median 6, max 17 |
| sporer | 8,107 | **0** | 4,311 | 5,421 | **≡ 1 for all** |
| rgo | 561 | 6,709 | 30,350 | 25,611 | median 5, max 22 |
| debrecen | 343 | 2,231 | 13,862 | 11,412 | median 6, max 41 |
| kmas | 1,319 | 3,320 | 22,024 | 22,295 | median 5 (clean) |

Two of Check A's five measures are built on spotless days, and Spörer has no multi-day groups to
give lifetimes, so the PI excluded both surveys from Step 1 on 2026-11-21. **Schwabe's handling is
deferred to Step 4**, where the MNAR reporting layer must face the pre-1874 era anyway. Whether
upstream collapsed their spotless days into no-observation, or never read them, is worth raising
with `sunspot-homogenization`.

**Consequence:** Step 1 produces no pre-1874 reference statistics. Check B spans 1874–2026.

### 1.5 Data refresh

SILSO SN V2.0 refreshed into `data/raw/silso_2026-11/`: daily to **2026-10-31** (was 2018-07-31)
and monthly to **2026-10** (was 2017-10), which resolves **D6** for SILSO. The files are byte-exact
as SILSO serves them — semicolon-separated and headerless, unlike the converted 2018 snapshot — and
the loader reads both. Comparison only, enforced by the config validator.

The published GN comparison series move to **Step 2**, where the benchmark consumes them (PI).

### 1.6 Issue statuses

All of **D1–D23** now carry a status in [`docs/04_data/known_issues.md`](../04_data/known_issues.md).
Resolved in this step or by v1.3_4: D6 (for SILSO), D16, D17, D18, D19 (partly), D20. Newly
registered: **D23**. Open by design: D5, D8, D10–D15, D21. Handled in code: D22, D23, and the
KMAS single-use rule.

## 2. Check B — is the area distribution stationary?

**Design.** Five epochs spanning 1874–2026, binned by activity quintile using the **91-day smoothed
group count** (PI's choice: conditioning on smoothed *area* would condition on a function of the
quantity under test). Primary test restricted to `area_corr ≥ 30 MSH` (D14). Significance from a
**27-day block bootstrap**, because groups within a day and days within a rotation are dependent.

**Result: stationary across the anchor-era epochs** (`f10`).

| epoch | groups ≥ 30 MSH | p10 | p50 | p90 | fraction ≥ 30 MSH |
|---|---|---|---|---|---|
| rgo_early (12–14) | 25,694 | 1.673 | 2.200 | 2.750 | **0.819** |
| rgo_anchor (15–19) | 66,447 | 1.658 | 2.203 | 2.783 | 0.724 |
| rgo_late (20) | 14,402 | 1.642 | 2.166 | 2.739 | 0.736 |
| debrecen_anchor (21–23) | 42,747 | 1.676 | 2.214 | 2.822 | 0.767 |
| kmas_bridge (20–24) | 66,442 | 1.653 | 2.208 | 2.784 | 0.747 |

- Percentile spread across all five epochs: **p10 0.034, p50 0.049, p90 0.083 dex** — inside the
  0.1 dex target.
- Pairwise within activity bins: max |Δp50| **0.079 dex**, max |Δp90| **0.097 dex**.
- 7 of 43 epoch-pair tests reject at p < 0.05, and **the rejections are concentrated in
  `rgo_early`** (KS 0.093–0.120 against every other epoch, against a median of 0.033 overall).
- The above-30-MSH fraction spans 0.724–0.819, a 13% relative spread — **outside** the 10% target.
  Dropping `rgo_early` leaves 0.724–0.767, a 5.9% spread, inside it.

**Reading.** The one epoch that fails is the one ADR-009 already excludes from the secondary anchor
on D5 grounds. The stationarity test therefore *rediscovers* early RGO's inhomogeneity from areas
alone, which is a useful independent check on a decision that was made on documentary grounds.

**Consequence for the detection layer.** Over the anchored epochs, `q_o = E_F[π_o(a)]` may be
computed against a single area law without an epoch effect. **Early RGO must carry its own**, which
is what treating each reference segment as its own observer already does (ADR-003, D5). Nothing
here licenses extrapolating the law to 1610 — counts cannot test that, and the methods paper has to
say so.

**Two-population structure: strongly supported** (`f11`). A two-component lognormal mixture beats a
one-component fit by ΔBIC **+965 to +4,268** in every epoch, with stable components: small
component at log₁₀ area ≈ 1.65 (weight 0.12–0.15), large at ≈ 2.30. Debrecen is the outlier
(small-component mean 1.82, weight 0.27), which is the completeness effect of D14 — it is the
deepest survey — not a solar difference.

## 3. Check A — do immigration–death dynamics reproduce the reference?

*(Filled in below from `outputs/step1/check_a/scores.csv`.)*

## 4. The forward filter's cost (Step 4, early task 1)

Measured, replacing the research report's unbenchmarked estimates (`f12`):

| states | T | CPU | one A100 |
|---|---|---|---|
| 41 | 10,000 | 0.042 s | 0.451 s |
| 41 | 40,000 | 0.169 s | 1.826 s |
| **41** | **152,000** | **0.667 s** | **6.950 s** |
| 1,681 | 10,000 | 0.272 s | 0.644 s |
| 1,681 | 40,000 | 1.093 s | 2.569 s |

The report estimated 0.2–1 s on CPU — **confirmed** — and 1–3 s single-chain on GPU, which is
**wrong by 2–7×**. The report's own reasoning explains it: a sequential scan with 41-vector bodies
is latency-bound, so per-step launch overhead dominates and the GPU loses by an order of magnitude.
It should win for batched chains, which is what Step 4's blocked Gibbs needs, but that is now a
thing to measure rather than assume.

**Acted on in this step:** Check A pins itself to the CPU by default, which is a tenfold saving on
the step's own compute.

## 5. Settings handed to Step 4

| Setting | Value | Why |
|---|---|---|
| Truncation **K** | **41** | Smallest K with `P(GROUPS ≥ K) < 1e-5` that exceeds the reference's largest daily count (31, KMAS). v1.3_4's rule minimum is 39, v1.3_3's was 41, so 41 is stable across both releases (`f03`) |
| Per-class K, two classes | 21 | 441 joint states; the sum still reaches 40 |
| Lifetime parametrization | geometric survival **times an explicit rotation kernel** | `obs_duration` is a disk-passage-censored *span*, not an intrinsic lifetime (`f06`). φ is a visible-residence parameter and must be documented as one |
| Area classes | *(see Check A)* | |
| Anchors | unchanged (ADR-009) | |
| Device for single-chain filtering | **CPU** | Measured, section 4 |

## 6. Reproducing this

```bash
conda activate gn-next-gen
pytest                      # fast suite
pytest -m slow              # includes the v1.3_3 regeneration and the recovery test
python scripts/diff_gn_releases.py
python scripts/check_a_dynamics.py --config configs/step1_check_a.yaml
python scripts/check_b_stationarity.py --config configs/step1_check_b.yaml
python scripts/benchmark_forward_filter.py
python scripts/make_step1_figures.py
```

Figures are committed under [`figures/step-1/`](figures/step-1/) in light and dark; the CSVs behind
them are in git-ignored `outputs/step1/`, regenerated by the commands above.
