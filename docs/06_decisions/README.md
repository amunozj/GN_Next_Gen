# Architecture decision records

Decisions are numbered and never rewritten; supersede them with a new ADR (template:
[../templates/adr.md](../templates/adr.md)).

| ADR | Decision |
|---|---|
| [001](ADR-001-stack-jax-numpyro-conda.md) | Python + JAX/NumPyro; conda envs, GPU default (WSL2/Linux), CPU variant |
| [002](ADR-002-two-variants-and-firewall.md) | Sunspot-only and proxy-informed variants; data-lineage firewall |
| [003](ADR-003-reference-equivalent-target.md) | Estimand = reference-equivalent count, one anchored reference epoch; no SILSO in calibration |
| [004](ADR-004-data-scope.md) | Counts + partial metadata + occasional blanket statements; no drawings/positions |
| [005](ADR-005-legacy-as-text-snapshot.md) | Legacy code as read-only text snapshot in `legacy/` |
| [006](ADR-006-data-immutability-and-storage.md) | `data/raw` immutable, plain git, checksum manifest |
| [007](ADR-007-benchmark-first-limiting-cases.md) | Blind benchmark before claims; existing methods as limiting cases |
| [008](ADR-008-deliverables.md) | Full GN series as posterior ensemble + methods paper; MM calibrated, not the focus |
| [009](ADR-009-reference-survey-and-anchored-epochs.md) | Reference = five per-survey catalogues (native areas); two co-equal anchors: Debrecen 21–23, RGO 15–19 |
