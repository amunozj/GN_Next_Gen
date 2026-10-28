# ADR-005: Legacy code is a read-only text snapshot inside the repo

- Status: Accepted (2026-10-28, PI)
- Context: Development moves to another machine; the repo must be self-contained, small, and readable by Claude Code.
- Decision: Copy all legacy `.py`/`.m` source, small run CSVs and comparison series into `legacy/`; export notebooks to
  py:percent text with outputs stripped; record provenance (remote, commit, dirty state, exclusions) per folder. Original
  repos stay untouched; several notebooks were untracked there and exist only on the original machine.
- Consequences: Legacy files are never edited. Ports into `src/` cite the legacy file. Figures from old notebooks are not
  available in-repo.
