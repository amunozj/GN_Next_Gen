# Results: Step 2 — a blind benchmark of synthetic observers

- Plan: [`docs/plans/step-2-plan.md`](../plans/step-2-plan.md) (Approved 2026-11-22)
- Spec: [`docs/05_roadmap/step-2-blind-benchmark.md`](../05_roadmap/step-2-blind-benchmark.md)
- Branch: `step-2-blind-benchmark`
- Environment: conda env **`gn-next-gen`** (Python 3.12, pandas 3.1.1, SciPy 1.18)
- Decisions taken in session by the PI, 2026-11-22: hybrid replay; every baseline
  ported and no legacy code run; ADF's cycle phase from the observations; six
  scenarios × 10 realizations; **seven** baselines, not two; four structural
  misspecifications; the `runcsvs/` reproduction dropped; the published series
  acquired; ten named case-study observers; **no run over ~15 minutes launched
  from a Claude Code session**

> ### Scope of what follows
>
> **The full grid has run**: 51 worlds across five scenarios, nine methods,
> **459 scored cells** plus a 405-cell structural ensemble, no stage failed.
> It took 16 minutes on 64 cores, not the 3–6 hours planned.
>
> That includes scenario **S6**, the structural ensemble: 405 cells, three knobs
> × five settings × three worlds × nine methods, the full planned version.

> ### ⚠ Status, 2026-11-24 — §§4–9 are awaiting a re-run
>
> Three defects found while answering questions about this document are **fixed in
> code but not yet in these numbers**: the yearly coverage aggregation (§12.7), the
> missing hard floor (§12.8), and the `clean-modern` case study (§12.9). All three
> change what the grid produces, so every number in **§§4–9, and f18–f23, f25 and
> f26, is the 2026-11-23 run** and is superseded. What is *not* affected: §3 (the
> A157 validation reads RGO directly, not the worlds), §10 (the published series
> are frozen raw data), and §11.
>
> The re-run is one command and about 20 minutes on 64 cores. It is **not launched
> from a Claude Code session** (plan decision A13), and it regenerates rather than
> resumes, because `clean-modern` moving reshuffles every world's draws:
>
> ```bash
> ./scripts/run_step2_grid.sh --detach          # regenerates, re-fits, re-scores, redraws
> ```
>
> Staleness is enforced rather than trusted: `config.json` now carries a
> `generator_version`, and `is_complete` refuses a world written by an older
> generator, while `cell_done` refuses a fit whose parquet lacks the new
> uncertainty columns. Nothing silently reuses the old grid. The plan behind all
> three is
> [`docs/plans/step-2-scorer-followups.md`](../plans/step-2-scorer-followups.md).

## Summary

| Question | Answer |
|---|---|
| Do the ports behave as the published assessment says? | **Yes in sign, all five**, on that paper's own setup — HoSc98 and SvSc16 underestimate more as acuity rises, CEA17, UEA21 and VHS24 stay near zero (§3). Their *exclusion limits* differ, for a measured reason: our renovated RGO carries fewer small groups |
| Are any method's uncertainties calibrated? | **No. Not one, in any scenario.** At a nominal 50% the seven deliver **0.009 to 0.159**; at 90%, 0.025 to 0.272 (§5) |
| Can the published methods reach the Maunder Minimum at all? | **Five of seven cannot.** Every overlap-based method stops at **1774** and covers 46% of the record. The two that need no overlap reach 1660–1672 and overestimate the Maunder level by **23–25×** (§6) |
| Does the benchmark separate skill from noise? | **Yes, and four methods fail it.** VHS24, CEA17, UEA21 and GRP all score worse on yearly centred RMSE than a random segmentation of the same data (§4) |
| Which observers break a method? | **Slow drift breaks all seven.** A documented break costs 40.1 MSH of acuity error against 111.5 for the same break undocumented — the metadata is worth a factor of 2.5 (§7) |
| What does the observing calendar alone cost? | **About 5% of the trend** — +0.014 against a true 0.287 groups/century (§9). The large biases here are detection and reporting, not sampling |

## 1. What the benchmark is

![Six stages: truth from real groups, synthetic observers, real calendars and overlap graph, a sealed truth, nine methods run blind, and scores at four time scales.](figures/step-2/f13_benchmark_design_light.png)

**Figure f13** — The benchmark end to end. The generator differs from the Step 4
model in four stated ways, so the benchmark cannot flatter the model it exists to
judge; the research notes are blunt that "benchmarks that reuse the same
generative model as the inference method will overstate performance".

**Truth is real data, resequenced.** Inside the donor survey's own span the truth
*is* the reference: over 1877–1969 it reproduces RGO's daily reference-equivalent
counts **exactly, on all 24,219 days**. Outside it, whole complete cycles are
shifted rigidly in time and thinned by deleting whole groups, so every group keeps
its real area profile and its real visible span.

![Top: the replayed truth 1610-2026 as yearly mean groups per day. Bottom: which real RGO cycle each era was drawn from.](figures/step-2/f14_replay_world_light.png)

**Figure f14** — One replayed world and its provenance. The lower panel shows
which real cycle each synthetic era came from. Cycle amplitudes come from a
seeded AR(1), never from SILSO, a proxy or a published series — the firewall
(ADR-002) requires it, and it is also what keeps the truth genuinely unknown.

Writing the fidelity test earned its keep: synthetic cycles tile past the end of a
gap by construction, so without a clip they spilled into the verbatim span and
counted the same Sun twice.

## 2. What makes it differ from the closest precedent

Chatzistergos, Krivova, Sundermann & Usoskin (2025, A&A 699, A157) is the nearest
published synthetic-observer test, and the assessment recorded it as "full text not
read; which method is unconfirmed". It has now been read.

| | A157 | here |
|---|---|---|
| Truth | RGO, **1900–1976 only** | 1610–2026, real groups, areas and lifetimes |
| Grand minimum | none | S3, by whole-group deletion |
| Observers | 121, each a **hard acuity threshold** | soft thresholds with drift, breaks, split/merge, seeing noise |
| Drift, breaks, conventions | **excluded**: "which we do not consider here" | all present, some documented as metadata, some not |
| Spotless days | as reported | MNAR under-reporting, activity-dependent |
| Lifetimes | none | real `obs_duration`, through disk passage |
| Sampling | days **deleted at random** | real historical calendars, transplanted |
| Overlap structure | each observer **pairwise to RGO** | the real overlap graph |
| Daisy-chaining | **"not evaluated in this study"** | the chained methods' error is what is measured |
| Uncertainty calibration | **none** | the load-bearing metric (ADR-007) |

![Left: effective resistance to the anchor against each observer's first year. Right: the distribution of overlapping days per edge.](figures/step-2/f15_overlap_graph_light.png)

**Figure f15** — The structure A157 does not have. Chained calibration's variance
grows like effective resistance to the anchor, so the left panel is the x-axis of
every chained method's error. The reference observer sees only its own anchored
epoch (ADR-009), so every other observer is reached through this graph. **39 of the
423 observers never reach it**: the graph is not one connected whole. These panels
summarise the full-record network (**S2**); f28 draws it.

![Left: the overlap graph over 1610-2026 drawn as a network, nodes placed at each observer's mid-career year and their effective resistance to the anchor, edges in grey and bridges in orange. Right: the number of edges crossing each year's cut, falling to zero in 1727-1745 and 1777-1798.](figures/step-2/f28_overlap_network_light.png)

**Figure f28** — The graph f15 summarises, drawn in full (**S2**, 1610–2026). Left:
every one of the 16,858 edges, each observer placed at its mid-career year and its
resistance to the anchor, so the chain is visible as the thing a method has to walk.
Median resistance rises from 1.4e-5 for observers centred after 1980 to 2.7e-3 before
1830; **20 edges are bridges** (orange — cut one and everything behind it loses the
anchor); **39 of 423 observers have no path to the anchor at all**. Right: how much of
the chain crosses each year, splitting the observers by career midpoint (the ends of
the record, where one side holds fewer than ten observers, are trivially empty and
left out). **In 1727–1745 and 1777–1798 nothing crosses at all** — 41 years where the
graph is severed, not merely thin — and across 1727–1801 the cut never exceeds five
edges and holds one or none in 57 of those 75 years. That is the structural reason
five of seven methods stop at 1774 (§6): a property of the real overlap pattern, not
of any method. Numbers from
[`data/step2_overlap_graph_cut.csv`](data/step2_overlap_graph_cut.csv).

![Acuity distribution with the ten case studies marked, MNAR omission rates, and counts of observers with breaks, copiers and blanket records.](figures/step-2/f16_observer_population_light.png)

**Figure f16** — The observer population. Everything in the third panel — career
breaks, copied records, blanket statements — is named and then set aside by A157's
own statement, and modelled here.

![Left: observing days against first year, per station. Right: observers by decade of first observation.](figures/step-2/f17_calendars_light.png)

**Figure f17** — The calendars, transplanted verbatim from the real record
including the sparse 17th- and 18th-century patterns where the graph is thinnest:
**44 observers before 1800**, a median of 2.5 per decade, two-thirds of them with
fewer than 300 observing days in total. A157
reduces overlap by deleting days at random instead, which cannot reproduce a
bottleneck.

## 3. Criterion A7 — do the ports behave as published? *(complete)*

The plan dropped the `legacy/ADFSSN/runcsvs/` reproduction: those files were not
produced by the committed driver settings, and the snapshot already carries
`uncommitted.diff`, so the target is not well defined. An external, published
target replaces it — rebuild A157's own setup and compare.

![Left: mean difference against acuity for five methods, HoSc98 and SvSc16 falling away. Right: RMS difference growing with acuity for all.](figures/step-2/f24_a157_validation_light.png)

**Figure f24** — A157's Fig. 3b, rebuilt with these ports on their setup (RGO
after 1900, hard thresholds, full overlap, annual means). **All five agree in
sign**: HoSc98 and SvSc16 fall away with acuity, CEA17, UEA21 and VHS24 stay near
zero. CEA17 lies exactly under UEA21, which is their result too — "by design,
CEA17 and UEA21 are identical in these metrics in this case".

| method | mean-difference slope | falls with acuity? | A157 expects | agrees |
|---|---|---|---|---|
| HoSc98 | −0.00157 | yes | yes | **✓** |
| SvSc16 | −0.00076 | yes | yes | **✓** |
| VHS24 | +0.00029 | no | no | **✓** |
| CEA17 | +0.00059 | no | no | **✓** |
| UEA21 | +0.00059 | no | no | **✓** |

**The exclusion limits do not reproduce, and the reason is measured.** HoSc98 gives
up at **73 msd** here against their **31**. Its factor is a ratio of sums, so
losing a fraction `f` of group-days implies `k ≈ 1/(1−f)`, and its own
`0.6 < k < 1.4` gate trips at `f = 0.29`:

| acuity | group-days lost | implied `k` |
|---|---|---|
| 31 msd | 15.0% | 1.18 |
| 60 msd | 26.7% | 1.36 |
| 87 msd | 33.3% | 1.50 |

Our renovated RGO crosses `f = 0.29` between 60 and 87 msd, which predicts 73
exactly. A157 used the 2017 catalogue, whose small-group completeness differs —
D14 puts RGO's own floor at 28.1 MSH. **This is a data difference, not a port
defect**, and the evidence is in `a157_catalogue.csv`.

Units are the trap here and are worth stating: A157 quotes acuity in millionths of
the solar *disc*, this project works in millionths of the *hemisphere*, and the
disc is half the hemisphere, so MSH = msd / 2 (D4).

## 4. Baseline scores *(full grid: 459 cells)*

> **Naming, 2026-11-24.** The port of the 2017 MATLAB multipath method is called
> **Pathfinding** here and everywhere in the code; it was called `Backbone` through
> 2026-11-23. The old name collided with **SvSc16**, which *is* Svalgaard &
> Schatten's backbone and is a separate baseline in the same table, so "the backbone
> method" named two of the seven. Nothing about the method, its inputs or its scores
> changed — the figures and CSVs carry the new label.

![Bias as a marker and centred RMSE as a bar, at daily, monthly and yearly scales, for nine methods, each method drawn in its own colour and marker shape.](figures/step-2/f18_baseline_scores_light.png)

**Figure f18** — Point error at three averaging scales, pooled over all 51 worlds.
A method can be unbiased daily and wrong per cycle, which is why HOME scores at
several scales and so does this. Each method keeps one colour **and** one marker
shape across f18–f26, so a mark says which method it belongs to without the
legend, and the two nulls stay apart in the grey they share.

| method | yearly bias | yearly cRMSE | CRPS | beats the null? |
|---|---|---|---|---|
| HoSc98 | −0.234 | **0.312** | 0.248 | yes |
| Pathfinding | −0.194 | 0.373 | 0.265 | yes |
| SvSc16 | −0.254 | 0.379 | 0.285 | yes |
| *null-segmentation* | *0.145* | *0.749* | *0.415* | — |
| VHS24 | **0.013** | 0.787 | 0.542 | **no** |
| CEA17 | 0.008 | 0.791 | 0.470 | **no** |
| UEA21 | 1.145 | 0.865 | 1.188 | **no** |
| GRP | 0.654 | 0.945 | 0.868 | **no** |
| *null-prior* | *0.274* | *2.447* | *1.907* | — |

**Four of the seven are beaten by a random segmentation.** Cutting careers at
arbitrary points and applying arbitrary scalings scores 0.749; VHS24, CEA17,
UEA21 and GRP are all worse. On the provisional two-world slice this was three
methods and CEA17 was ahead of the null — the full grid moves it the other way.
Lindau & Venema (2018) found that at realistic signal-to-noise a random
segmentation explains about half the break variance, which is exactly why the
null is in the table. A ranking without it would have presented four methods as
working.

Note the two columns disagree about who is good. VHS24 and CEA17 have the
*smallest* yearly bias of any method (+0.013, +0.008) and among the worst
scatter. A method can be centred and useless.

![Horizontal bars of the slope of estimated-minus-true cycle amplitude against true amplitude, per method.](figures/step-2/f19_cycle_amplitude_light.png)

**Figure f19** — The secular-bias signature on the real overlap graph. The two
families separate cleanly: the linear scalings and Pathfinding sit at
**−0.006 to −0.017**, effectively flat, while CEA17, UEA21, VHS24 and GRP run
**−0.21 to −0.38**.

**This is a partial reproduction, and the part that moved should be said
plainly.** On their own setup (§3) all five methods agree in sign with the
published Fig. 3b. On the *real overlap graph* the separation survives but the
linear group's sign does not: A157 finds HoSc98 and SvSc16 amplifying strong
cycles, and here they are flat to very slightly negative. The two-world slice
reported in an earlier draft of this document gave them +0.008 and +0.001, which
the full grid does not support either. What reproduces robustly is the
*separation* between the families, not the sign of the linear one.

## 5. Coverage — the finding this step exists for

![Empirical against nominal coverage at 50, 68, 90 and 95 percent; every method far below the diagonal.](figures/step-2/f20_interval_coverage_light.png)

**Figure f20** — Nominal against empirical interval coverage, over the 450
calibration cells. **Every curve lies far below the diagonal.**

**Read these levels as a lower bound**: the yearly aggregation behind them treats a
year's daily errors as independent, which they are not, and §12.7 gives the size of
that effect and which rows survive it.

| nominal | best of the seven | worst of the seven |
|---|---|---|
| 50% | 0.159 (Pathfinding) | 0.009 (UEA21) |
| 68% | 0.199 (SvSc16) | 0.014 (UEA21) |
| 90% | 0.272 (Pathfinding) | 0.025 (UEA21) |
| 95% | 0.301 (Pathfinding) | 0.030 (UEA21) |

S5 is excluded from this figure and table. With detection error switched off an
exact method scores a coverage of 1.0 there, so it is a check on the pipeline
(§9) rather than a test of calibration, and letting one degenerate world carry a
fifth of the headline would flatter the two scaling methods.

The assessment's charge — "no evidence that any method's uncertainties are
calibrated" — is not merely unaddressed in the literature. On this benchmark it
is false in the strongest possible direction, for every method, in every
scenario. The numbers barely move between S1, S2, S3 and S4 (0.13–0.18 at
nominal 0.50 for HoSc98 across S1–S4), so this is not a quirk of one world or
one era.

This is the axis on which the comparison is new. A157 scores RMS, mean, relative
and maximum-absolute difference, and scores no interval at all. CEA17 is the only
baseline whose construction yields a native distribution — "the calibration
returns a PMF series, which also includes information about the uncertainty" —
and its interval covers 0.050 of truth at a nominal 0.50.

One honest caveat: the nulls are badly calibrated too, so this figure alone does
not separate "overconfident" from "wrongly centred". The PIT statistics in
`scores.csv` do, and they are large (0.43–0.91) for every method.

## 6. Reach: five of seven methods cannot see the Maunder Minimum

The result the two-world S1 slice could not show, because S1 begins in 1874.

| method | earliest year reached | share of the 1610–2026 record |
|---|---|---|
| HoSc98, SvSc16, VHS24, CEA17, Pathfinding | **1774** | **46%** |
| UEA21 | 1672 | 56% |
| GRP | 1660 | 58% |

Every method that calibrates through observer **overlap** stops at 1774. The
reference observer sees only its anchored epoch (ADR-009: RGO cycles 15–19,
1912–1969), and the chain from there does not reach the 17th century: the graph
is too sparse. These methods do not produce a wrong Maunder Minimum — they
produce **none at all**, over more than half the record the series is supposed
to cover.

The two methods that need no overlap do reach it, and are wrong by more than an
order of magnitude. On S3, where a grand minimum is inserted by deleting whole
groups:

| method | true Maunder level | estimated | ratio |
|---|---|---|---|
| UEA21 | 0.088 | 2.165 | **24.5×** |
| GRP | 0.094 | 2.135 | **22.8×** |
| *null-segmentation* | *0.090* | *0.109* | *1.2×* |

**A random segmentation gets the Maunder level right to 20% while both real
methods overestimate it twenty-fold.** That is not because the null is clever: it
rescales raw counts, which in a grand minimum are near zero, so it inherits the
right answer. It is in the table because on the single number that decides the
Maunder Minimum, neither method that can reach the era beats it.

Taken together these two findings are the quantitative case for this project:
the published methods either cannot speak about the Maunder Minimum, or speak
about it wrongly by a factor of twenty.

## 7. The ten case-study observers

An aggregate score says which method wins and hides the observers that break one.
Ten observers are fixed by name and seed, each isolating a failure mode; 3,500
case-study rows across S1–S4, which is **fifty replicates** — scenario × seed —
behind every observer and method. The medians, quartiles and ranges below are in
[`data/step2_case_study_observers.csv`](data/step2_case_study_observers.csv), the
same table `f25` and `f26` are drawn from.

Fifty replicates make two different questions available, and they give different
answers. A method is **typically** off when its *median* scale factor sits more
than 2× from the truth; it is **sometimes** off when its *worst* replicate does.
Reading only one of the two is how a case study gets misread:

| Observer | typically >2× off | sometimes >2× off | never answered |
|---|---|---|---|
| `clean-modern` | **4** | 4 | SvSc16 |
| `sparse-maunder` | 0 | 1 | UEA21 |
| `break-undocumented` | 1 | 2 | — |
| `break-documented` | 0 | 1 | — |
| `slow-drift` | 0 | **7** | — |
| `poor-instrument` | 1 | 4 | SvSc16 |
| `leaf-node` | 1 | 3 | UEA21 |
| `copier` | 0 | 4 | — |
| `zero-omitter` | 0 | 1 | — |
| `blanket-statements` | — | — | all seven |

![Five panels for case studies 1 to 5, each with recovered acuity above and recovered-over-implied scale factor below; every method shows its median over fifty replicates as its own marker shape, its interquartile range as a thick bar and its full range as a thin one.](figures/step-2/f25_case_studies_easy_light.png)

**Figure f25** — Case studies 1–5. **`clean-modern` is the worst result in the
figure**, which is the whole point of including it: the case every method should
get right puts four of seven more than 2× off *in the median* — GRP 4.6×, UEA21
4.6×, VHS24 3.0×, CEA17 2.2× — and the two methods that estimate an acuity return
a median 300 MSH against a true 10. SvSc16 returns nothing for this observer in
any of the fifty replicates, and no method produces a usable scale factor in all
fifty.

**`slow-drift` is the opposite shape, and the ranges are what tell them apart.**
Every median sits within 1.9× of the truth, so on a median-only reading a random
walk in acuity defeats nobody; but every one of the seven has at least one
replicate past 2×, HoSc98 by 6.2× and Pathfinding by 5.4×. Drift does not bias these
methods, it makes them unreliable — a claim no single replicate, and no median,
can carry on its own.

The paired breaks still price the metadata, in acuity rather than scale: the
**same** break is recovered at a median **50 MSH when documented** and **160 MSH
when not** by UEA21, against a true 45, and at 28 against 58 by GRP. Documentation
is worth a factor of 2–3 here.

![Five panels for the hard cases — poor-instrument, leaf-node, copier, zero-omitter and blanket-statements — with recovered acuity above and scale factor below, each method drawn as a median marker in its own shape with interquartile and full ranges.](figures/step-2/f26_case_studies_hard_light.png)

**Figure f26** — Case studies 6–10. `leaf-node` and `poor-instrument` are the two
hard cases with a *typical* failure rather than an occasional one: GRP sits 2.6×
and 2.5× off in the median. Both also show the other kind of failure, the one the
counts under each slot record — **UEA21 returns nothing at all for `leaf-node`,
in all fifty replicates**, and SvSc16 nothing for `poor-instrument`. A method that
cannot answer is not the same as one that answers badly, and an aggregate score
drops the first silently.

`copier`, whose record duplicates a neighbour's, has no median past 2× but four
methods with replicates that are, and UEA21's acuity spans 30–300 MSH against
a true 30: correlated observer error, which nothing under test models, shows up as
spread rather than bias. `zero-omitter` is the one clear success — UEA21 recovers a
median 26 MSH against a true 30 and only CEA17 ever strays past 2× — so
activity-dependent omission of blank days is the failure mode these methods *do*
survive. `blanket-statements` produces nothing from any method in any replicate,
the zero under every slot, because the record is sentences rather than counts.

![Recovered against true acuity threshold, for the two methods that estimate one.](figures/step-2/f23_acuity_recovery_light.png)

**Figure f23** — Acuity recovery, for the only two methods that attempt it.
Nearly every point sits above the diagonal: observers are judged worse than they
were, by **+89 MSH (UEA21)** and **+87 MSH (GRP)** on average. A157 reports
exactly this direction for UEA21 — it "systematically overestimates the acuity of
the observer, thus in the end also amplifies the group numbers" — and the
benchmark reproduces it at scale.

## 8. The structural ensemble

![Bars spanning each method's yearly bias across its own free graph choices, on one S2 world.](figures/step-2/f22_structural_ensemble_light.png)

**Figure f22** — The spread a method's *own* free choices produce. Each bar spans
one baseline re-run across the overlap graph's day tolerance, minimum overlap and
minimum distinct counts — three knobs × five settings — over three S2 worlds and
all nine methods, the full planned ensemble at **405 cells**. Williams et al.
(2012) ran perturbed-parameter ensembles of the PHA for exactly this reason.

**The ensemble is narrower than the uncertainty the methods already report**, for
every one of them:

| method | structural spread | mean reported sd | wider? |
|---|---|---|---|
| HoSc98 | 0.192 | 0.645 | no |
| Pathfinding | 0.221 | 0.669 | no |
| SvSc16 | 0.153 | 0.736 | no |
| CEA17 | 0.401 | 0.510 | no |
| UEA21 | 0.229 | 0.713 | no |
| GRP | 0.372 | 0.550 | no |
| VHS24 | 0.508 | 1.084 | no |

That is a sharper statement of §5 than coverage alone. These intervals are not
too narrow because the methods ignored their own free choices — the choices move
the answer by less than the interval already allows. They are too narrow because
what the interval is a width *of* does not include the error that actually
dominates. Widening an interval to admit structural choices would not fix the
coverage; the error is elsewhere.

The figure withholds the default-setting marker because §4's defaults pool all
scenarios while this ensemble ran on S2 alone, and plotting the two together
would compare scenarios rather than settings.

## 9. The Hay sampling-only experiment

Scenario S5 puts **perfect** observers — no threshold, no drift, no noise, no MNAR
omission — on the real calendars. Every observer reports the estimand exactly on
the days it looked, so any residual error is the sampling pattern and nothing
else.

S5 also turns out to be a correctness check on the whole pipeline: on it,
**HoSc98 and Pathfinding score a centred RMSE of exactly 0.000 and a coverage of
1.000**. When there is nothing to calibrate, the calibrators return the truth.
That the two scaling methods do and the four non-parametric ones do not (0.86 to
1.22) is itself informative about where their error comes from.

![Left: induced trend bias against network size, with scatter narrowing. Right: the induced bias in three windows.](figures/step-2/f21_sampling_only_light.png)

**Figure f21** — What the calendar alone costs. The induced trend bias is
**+0.014 groups per century against a true 0.287** over 1610–present — about 5% —
falling to +0.0001 over 1850–present where the network is dense. Bootstrapping
over 5 to 400 observers, the scatter narrows from ±0.21 to ±0.008 while the mean
bias stays near +0.01 to +0.02.

**This differs from the sea-level precedent, and the difference is the point.**
Hay et al. found sparse, gappy sampling moved the 20th-century trend by about
30%; here the calendar alone moves it by 5%. S5 has no *preferential* sampling by
construction — MNAR omission is switched off — so the comparison says the large
biases in the GN problem come from detection and reporting, not from who looked
when.

## 10. The eight published series, compared

![Top: eight published group-number series over 1610-2022, smoothed over one solar cycle and drawn on a log axis, converging after 1750 and fanning out across two decades of magnitude inside the shaded Maunder Minimum. Middle: each series' mean group count in the Maunder window as a filled marker and in the modern overlap as an open one, on a log axis, the Maunder markers spanning 0.05 to 3.28. Bottom: the difference between the 2024 Hoyt and Schatten update divided by 12.08 and the 1998 series it updates, within a thousandth of a group.](figures/step-2/f27_published_series_light.png)

**Figure f27** — Every published group-number series this project holds, on one
axis. Panel (a) is smoothed over one cycle and drawn on a log axis deliberately:
at annual resolution eight cycle-resolved series are a hairball that hides the slow
disagreement, which is the only thing worth comparing. Panel (b) is the comparison
itself; panel (c) is the check that licenses the single scale (**D24**) — the
supplement publishes its Hoyt & Schatten update in 12.08-scaled units while its own
reconstruction is in counts, and dividing by 12.08 reproduces the 1998 series on all
386 overlapping years to within 9×10⁻⁴. A series left unscaled would sit twelve
times too high and still look like a plausible solar cycle.

**Where they agree, they agree well.** Over **1749–1995**, where all six full-record
series overlap, the mean daily group count runs from **3.83 to 4.80** — a spread of
**25%**, and most of that is one series:

| series | 1749–1995 | 1645–1715 |
|---|---|---|
| Velasco Herrera 2024 (H&S update) | 3.83 | **0.055** |
| Hoyt & Schatten 1998 | 3.83 | **0.055** |
| Usoskin et al. (ADF family) | 4.18 | — |
| Svalgaard & Schatten 2016 | 4.45 | **1.01** |
| Chatzistergos et al. 2017 | 4.51 | — |
| Velasco Herrera et al. 2024 | 4.80 | **0.10** |
| Vaquero et al. *(window only)* | — | **0.20** |
| Zolotova & Ponyavin *(window only)* | — | **3.28** |

**Where they disagree, they disagree by sixty.** Over the Maunder Minimum
**1645–1715** the same quantity runs from **0.055 to 3.28 groups per day**, a factor
of **60**. Two of the eight are window studies rather than reconstructions, and
including them is the point: the disagreement is not a tail of one method's error
bar, it is two incompatible pictures of the same seventy years. Note also what is
*missing* from that column: the Usoskin and Chatzistergos series begin in 1749 and
1739, so neither says anything about the window at all. Four series, two of them
window studies, are the entire published basis for the Maunder Minimum's level.

**This benchmark explains the size of that gap.** Five of the eight series were
produced by methods scored here — HoSc98, SvSc16, CEA17, UEA21 and VHS24 — so the
scores in §§4–7 are statements about these series' provenance, not about
look-alikes. And the disagreement sits exactly where the benchmark finds no skill:
§6 measures that five of seven methods never reach the era at all, and that the two
that do overestimate the Maunder level by **23–25×** on a known truth (that figure
is from the 2026-11-23 grid; see the status box). A published spread of 60× across
methods whose Maunder-era error is of that order is not a puzzle; it
is the expected consequence, and it is the strongest argument in this document for
why the series needs rebuilding rather than re-weighting.

**One correction to the frozen table, recorded here because it changed numbers.**
The 2017 MATLAB snapshot's yearly files use `gn = -1` with `n_observations = 0` for
a year with no data, and the first export carried it through as a negative group
count: six years of Hoyt & Schatten and one of the Usoskin file. They are null now
(**D25**). Comparison-only data, so no score moves.

**A documented rule was not actually enforced, and now is.** That folder's
README claimed the validator refuses these as a calibration input. It did
not: `FIREWALLED_KINDS` is `silso`, `isotope`, `aurora`, `proxy`, and a GN
series would have been declared `kind: sunspot` and passed. This mattered
more than it had before, because what arrived with the supplement includes a
**daily** reconstruction over the same observers — the one shape that could
enter a likelihood looking like data. `validate_run_config` now refuses a
`kind: published_series` dataset in role `calibration`, and every
`lineage.json` carries the matching evidence.

It is a **separate rule from the ADR-002 firewall**, not another entry in
`FIREWALLED_KINDS`, for two reasons. It binds *both* variants — the
proxy-informed variant may calibrate on isotopes, but neither variant may
calibrate on someone else's recalibration of these observers. And a published
GN series *is* sunspot data: listing it as proxy-derived would have made each
run's own firewall statement false. The lineage record therefore gained
`published_series_calibration_inputs` and a `circularity_statement` beside
the existing firewall fields, and `tests/test_config.py` pins that the two
stay apart. If you want this recorded as a decision rather than as
enforcement of ADR-003, an ADR amendment is the place — it is not
relitigating anything, since the project had already claimed the rule in
three documents.

The supplement also carries a corrected raw observation database, 1610–**2022**,
in the v1.22 shape. It is not adopted: ADR-010 makes `gn_database_v1.3` release
`_4` the database of record, and those corrections are an author's judgement to
be compared against v1.3's changelog rather than merged. It is noted because it
is the only thing in hand that runs past v1.3's 2010 end.

## 11. Reproducing this

Checks, all quick:

```bash
conda activate gn-next-gen
pytest && pytest -m slow
ruff check . && ruff format --check .
python scripts/fetch_published_gn_series.py --check
python scripts/generate_benchmark.py --config configs/step2_benchmark.yaml \
       --scenarios S1 --realizations 2 --check     # determinism: digests match
python scripts/validate_against_a157.py            # §3, 16 s
```

The grid itself is **one command**:

```bash
cd /d0/amunozj/git_repos/GN_Next_Gen
./scripts/run_step2_grid.sh --detach
```

It returns immediately, survives logout, and does the stages in the only order
that is correct. Watch it with:

```bash
./scripts/run_step2_grid.sh --status
```

Each stage stamps its progress file with the invocation's run id, so `--status`
marks rows left by a different invocation as `other run` instead of mixing them
into this one's. Without that, a leftover 18-cell slice read as if it were the
459-cell grid.

**Safe to re-run.** Every stage skips cells already on disk, so an interrupted
grid resumes rather than restarting; if the machine reboots mid-run, start the
same command again.

**Measured, not estimated.** The full grid ran on 2026-11-22 in **16 minutes**
with `--jobs 16` on 64 cores: the 51 worlds were already on disk and skipped, the
459 baseline cells took 15.4 minutes and the 405-cell structural ensemble 14.4,
concurrently. The 3–6 hour figure in earlier drafts was a serial estimate and was
far too pessimistic.

**What it does, and why in that order.** One dependency is hard: nothing that
reads a world may start before `generate_benchmark.py` finishes, because the
runners discover work by globbing for *complete* realizations and one started
early would quietly process a subset and look finished. The script enforces that
barrier and aborts if generation fails, since everything downstream would
otherwise produce confident nonsense.

| stage | needs | runs with | ~time |
|---|---|---|---|
| `validate_against_a157` | nothing — reads `data/raw` only | — | 16 s |
| `generate_benchmark` | — | — **barrier** | 30–60 min |
| `run_baselines` | generation | the next two | 3–6 h |
| `run_baselines --structural` | generation (S2) | the other two | 1–2 h |
| `sampling_only_experiment` | generation (S5) | the other two | ~20 min |
| `score_benchmark` | baselines | `--structural` scoring | ~10 min |
| `score_benchmark --structural` | the ensemble | plain scoring | ~10 min |
| `make_step2_figures` | everything | — | < 3 min |

The three middle stages write to disjoint paths — the structural runner tags its
directories `seed-NNNN__knob=value` — and the two scoring passes write different
CSVs, so each group is safe to run at once. Defaults to `--jobs 16` per runner,
comfortable on this machine's 64 cores; a worker holds well under 2 GB.

A stage that fails does not abort the ones that do not depend on it. The run ends
with a summary naming every failure and a non-zero exit code.

Useful flags: `--jobs N`, `--scenarios S1 S2`, `--realizations N` and
`--methods HoSc98 CEA17` for a smoke run; `--help` for the rest. The underlying
scripts can still be run individually — the orchestrator only sequences them.

Figures are committed under [`figures/step-2/`](figures/step-2/) in light and dark, and
the per-observer summary behind §7 under
[`data/step2_case_study_observers.csv`](data/step2_case_study_observers.csv); the rest of
the CSVs are in git-ignored `outputs/step2/`.
## 12. What this step does not establish

Stated plainly, because a benchmark that overclaims is worse than none.

1. **This is not blind in the HOME sense.** HOME had 25 outside groups submit
   before the inserted inhomogeneities were revealed. One developer wrote both
   sides here. Blindness is enforced by API and audited by a static import test,
   and the baselines were frozen by commit before the scoring run. That is
   weaker than HOME and stronger than nothing.
2. **Grand-minimum truths are thinned modern groups, not a regime change.** No
   source was found for generating grand-minimum synthetic truths; the thinning
   is this project's choice. §6's 23–25× overestimate is conditional on it —
   though the *reach* result, that five methods never arrive at all, does not
   depend on what the truth looks like once you get there.
3. **Overconfidence is not decomposed.** There is no matched-model control arm,
   so §5 measures coverage under *this* generator rather than separating model
   error from data error.
4. **Break-detection skill is NaN for all seven.** None of them attempts to
   detect a career break. The metric is implemented and tested so Steps 3–4 can
   be scored on it; the absence is a result, not a gap in the scorer.
5. **One anchoring, not two.** Every scenario here uses ADR-009's secondary
   anchor (RGO cycles 15–19). The Debrecen 21–23 anchoring is supported by the
   code and configured per scenario, but was not run, so the anchor-sensitivity
   comparison ADR-003 asks for is still owed by Steps 3–4.
6. **The second, model-based generator was not written.** The replay generator
   already supplies the four structural misspecifications through real data; a
   parametric sibling would have added a knob without adding a departure from
   the Step 4 model.
7. **The yearly coverage behind §5 was aggregated wrongly, and is now fixed in
   code.** `uncertainty_scores` pooled a method's daily `gn_sd` to the year as
   `sqrt(mean(sd^2) / n_days)` — the independent-days rule. The dominant term in a
   chained method's `sd` is the observer's calibration factor error, which is
   *identical on every day of that observer's career* and does not average down, so
   the rule shrank every yearly interval by up to `sqrt(365) ≈ 19` and reported an
   overconfidence no method had claimed. Measured on 3 S2 worlds before the fix,
   carrying the factor error as correlated instead moves nominal-50% coverage from
   0.177 to 0.810 (HoSc98), 0.161 to 0.824 (Pathfinding), 0.145 to 0.730 (SvSc16),
   0.064 to 0.781 (VHS24) and 0.064 to 0.613 (CEA17) — **a sign change in §5's
   finding**. It does *not* move the conclusion for UEA21 (0.025 to 0.116) or GRP
   (0.014 to 0.246), which undercover under either rule. Each baseline now returns
   `gn_sd_factor` and `gn_sd_spread` beside an unchanged `gn_sd`, and the scorer
   pools them as `sqrt(mean(sd_factor)^2 + mean(sd_spread^2) / n_days)`. The test
   that was supposed to guard this could not: it sets one day per year, so the
   aggregation was a no-op; two tests with 200-day years now pin both halves.
   **§5's numbers below are the pre-fix ones** and are superseded by the re-run.
8. **The floor §4 compares against was the weak one, and two harder floors are now
   in the grid.** `null-segmentation`'s random `exp(N(0, 0.3))` factors average out
   over ~450 observers and a year of days, so it is the naive mean plus noise. A
   plain daily **median of the raw, uncalibrated counts** is a much harder floor: on
   3 S2 worlds it scores a yearly cRMSE of **0.391** against HoSc98's 0.343,
   Pathfinding's 0.448, SvSc16's 0.479 and `null-segmentation`'s 0.711 — **six of
   the seven methods lose to the median of the counts they were meant to
   calibrate**, and only HoSc98 beats it. `naive-mean` and `naive-median` are now
   registered floors; they are shape comparators only, with no anchoring and no
   uncertainty, so they appear on centred RMSE and nowhere else. §4's "four of the
   seven are beaten by a random segmentation" is true but understated, and its
   "beats the null?" column — a hand comparison of two pooled means — is now a
   per-world paired test written to `floors.csv` by the scorer.
9. **`clean-modern` was testing a short record, not a clean one.** The case study
   was selected as the densest post-1959 calendar, which is station 722: a
   **517-day career, 1992-01 to 1993-05**, about 17 of the fixed months UEA21's
   statistic is defined on against its own floor of 12. All fifty replicates reuse
   that one window, and the replay drops a different synthetic Sun into it each
   time. What the acuity methods then recovered was the window, not the observer:
   across the ten S2 worlds the recovered acuity tracks the window's true mean
   group count with **Spearman ρ = −0.93** (p = 1e-4), and in the one world whose
   window is more active than the anchor epoch UEA21 returns **12 MSH against a
   true 10**. The mechanism is the methods' own and is published — the ADF
   fingerprint cannot separate a quiet Sun from a blind observer, the library is
   built once over the anchor epoch with no conditioning on activity, and a
   quiet-era observer's statistic lies below every library entry, so the fit runs to
   the top of the grid and pins at the 300 MSH cap A157 documents. Selection now
   requires **≥ 3,000 observing days**, which moves `clean-modern` to station 459
   (6,922 days, 1977–1995, about two cycles), and f23 gained a panel showing
   recovered acuity against era activity. **§7's numbers below are the pre-change
   ones.**

## 13. What Steps 3–4 inherit

| Setting | Value | Why |
|---|---|---|
| Benchmark API | `gn_next_gen.benchmark` + `gn_next_gen.baselines`; one `calibrate(...) -> CalibrationResult` signature | A Step 3 or 4 method is scored by registering it in `BASELINES` |
| One command | `./scripts/run_step2_grid.sh --detach` | 51 worlds, 459 cells, 16 min on 64 cores; resumable |
| The bar to clear | yearly cRMSE **0.312** (HoSc98) and a 50% coverage of **0.159** (Pathfinding) | The best point error and the best coverage among the seven |
| The floor | a random segmentation scores **0.749** | Four published methods are below it; a method that is not has demonstrated nothing |
| The open goal | **no method reaches before 1774**, and the two that do overestimate the Maunder level 23–25× | This is the gap a joint model exists to close, now measured rather than asserted |
| Hardest observer | `slow-drift` — beats all seven | A drift prior is not optional |
| Coverage levels | 50 / 68 / 90 / 95, stratified by era | Reconciles the spec's 50–90 with the research notes' 68–95 at no cost |
| Overlap graph | `graph.overlap`, day tolerance 2, min overlap 60 days, min 4 distinct counts | 507 usable stations, 19,655 edges, one 422-node giant component, 26 bridges |
| Estimand | the anchor survey's own completeness threshold, declared per scenario | ADR-003 and ADR-009; the scorer never compares two anchorings |
| Known trap | composing PMF matrices leaks mass and biases the chained mean **low** | `pmf.compose` renormalises; any later matrix chaining must too |
