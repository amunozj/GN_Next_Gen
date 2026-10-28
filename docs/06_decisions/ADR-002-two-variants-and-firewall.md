# ADR-002: Sunspot-only and proxy-informed variants, separated by a data-lineage firewall

- Status: Accepted (2026-10-28, PI)
- Context: A GN series is used to test cosmogenic-isotope and other proxies; using those proxies to build it makes such
  tests circular. Proxies are, however, the only strong low-frequency constraint in the Maunder Minimum.
- Decision: Produce both a **sunspot-only** series and a **proxy-informed** series (¹⁴C/¹⁰Be, aurorae, optionally
  geomagnetic indices).
- Firewall rules for the sunspot-only variant: no isotope- or proxy-derived priors, hyperparameters, grand-minimum
  priors, cycle-amplitude priors, or threshold populations; no SILSO SN (which is itself a composite) in calibration.
  Its priors come from sunspot-era data or dynamo physics only. Data lineage is recorded per run.
- Proxy variant: published with leave-one-proxy-type-out siblings and a per-year prior→posterior contraction map.
- Consequences: Configuration must make proxy terms switchable and record which inputs entered each run.
