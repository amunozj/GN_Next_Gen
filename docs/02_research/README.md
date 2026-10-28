# Cross-domain research (2026-10-28)

How fields **outside** heliophysics solve problems structurally like GN homogenization, produced by a six-track
literature search during the ADFSSN modernization assessment. The heliophysics-specific literature review is in
[../01_assessment/assessment.md](../01_assessment/assessment.md).

| File | Content |
|---|---|
| [adjacent-methods-report.md](adjacent-methods-report.md) | Synthesis report (~100 KB). Start here. Ends with **11 transferable ideas ranked by payoff** and the "existing methods are limiting cases" framing |
| [notes/ecology_imperfect_detection.md](notes/ecology_imperfect_detection.md) | N-mixture (static, Dail-Madsen), occupancy incl. false positives, BBS observer/novice/aging effects, eBird expertise, integrated population models, size-biased detection |
| [notes/counting_processes_underreporting.md](notes/counting_processes_underreporting.md) | INAR/binomial thinning, M/G/∞ inference from occupancy counts, mark-dependent thinning, epidemiological under-reporting, MNAR zeros, capture-recapture bounds |
| [notes/raters_linking_ratings.md](notes/raters_linking_ratings.md) | Dawid-Skene, hierarchical rater models and rater drift, test equating (chained vs concurrent, multiple links), TrueSkill Through Time / Whole-History Rating |
| [notes/intercalibration_homogenization.md](notes/intercalibration_homogenization.md) | SDSS ubercal and self-calibration, satellite record merging and ensembles, TSI composites, sea-level reconstruction, station homogenization benchmarks (HOME) |
| [notes/paleoclimate_historical_evidence.md](notes/paleoclimate_historical_evidence.md) | BARCAST, LMR data assimilation and proxy system models, likelihoods for absence/qualitative statements, sunspot–isotope linkage, circularity firewalls, pseudo-proxy benchmarks |
| [notes/sbi_and_scalable_inference.md](notes/sbi_and_scalable_inference.md) | Neural SBI as an ADF replacement, hierarchical SBI, misspecification, SBC/TARP; exact forward filtering in JAX, parallel scans, blocked Gibbs, SVI, tempered SMC, cost estimates |

## How to read the evidence (important)

- A sentence with a linked source reports what that source says.
- *(derived)* and paragraphs opening with **Researchers' derivation** are the research team's own mappings, equations or
  estimates, **not** published results. Validate them before relying on them.
- **[nv]** marks references cited from domain knowledge and **not re-verified**. Check the DOI and the claim before
  citing in the paper.
- **(secondary)** marks claims checked only through a search summary or a citing source.
- All runtime, memory and flop figures are back-of-envelope estimates, **not benchmarks**. Step 4 must benchmark them early.
- A few primary PDFs could not be fetched during the search (noted in each file's *Gaps* sections).

## Sections most relevant to each roadmap step

| Step | Report sections (by heading) |
|---|---|
| 1 Data foundation & viability | "Integer autoregressions give lifetimes…", "M/G/∞ inference…", "Mark-dependent thinning…", "Size-biased detection…", "Two empirical checks" paragraph at the end of "A 41-state forward filter…" |
| 2 Blind benchmark | "Station homogenization supplies benchmarks…", "Circularity firewalls and pseudo-proxy benchmarks", "Sea level shows the sampling pattern itself biases trends", ranked idea 4 |
| 3 Global network calibration | "Ubercal: simultaneous solution…", "Equating quantifies chain drift…", "Whole-history ratings…", "Dawid-Skene generalizes…", ranked ideas 1 and 7 |
| 4 Joint latent-population model | "Ecology already wrote the GN likelihood…", "Epidemiology's under-reporting models…", "Spotless days are a selection problem…", "Absence of evidence is data…", "A 41-state forward filter…", "Samplers…", ranked ideas 2, 3, 5, 6, 8 |
| 5 Neural emission calibrator | "Simulation-based inference upgrades ADF…" (all subsections), ranked idea 9 |
| 6 Proxy-informed variant | "Isotopes and aurorae…", "Circularity firewalls…", "Integrated models join counts…", ranked idea 10 |
