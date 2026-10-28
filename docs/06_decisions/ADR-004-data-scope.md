# ADR-004: Data scope — counts, partial metadata, occasional blanket statements; no positions

- Status: Accepted (2026-10-28, PI)
- Decision: Inputs are (1) daily group counts per observer (SILSO group-number database), (2) the reference daily group
  areas, (3) observer metadata (telescope aperture, projection vs direct viewing, career dates, age) **where available —
  not for all observers**, (4) blanket qualitative statements ("no spots seen in year X") **only sometimes available**,
  and (5) proxies only in the proxy-informed variant (ADR-002).
- Out of scope: sunspot drawings, group positions/latitudes, butterfly data (a local collection exists on the original
  machine under `Codes/Bfly Diagram Update/` if this is revisited).
- Consequences: Metadata enter as optional covariates with missing values handled hierarchically; blanket statements are
  optional likelihood terms; capture-recapture with matched drawings (research idea 11) is not pursued.
