# ADR-003: The estimand is the reference-equivalent group count, anchored to one reference epoch

- Status: Accepted (2026-10-28, from the assessment)
- Context: Counts alone identify only relative abundance (N-mixture theory; SDSS ubercal's exactly degenerate zero
  point). Every reconstruction implicitly picks a scale.
- Decision: GN(t) is the number of groups a fixed **reference observer** would have counted on day t. One reference
  epoch (to be chosen in Step 1, e.g. a homogeneous RGO or Debrecen segment) has fixed detection parameters; other
  reference segments (early RGO, Debrecen, Kislovodsk) are treated as observers with their own parameters.
- Consequences: All observer parameters are relative to that epoch; the choice and its sensitivity are reported in the
  paper. SILSO SN is not used to define cycle phases or activity levels (cycle phase is latent in the model).
